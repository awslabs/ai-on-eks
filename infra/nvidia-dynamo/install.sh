#!/bin/bash

#---------------------------------------------------------------
# NVIDIA Dynamo on EKS - Installation Script
#
# This script sets up the complete Dynamo Cloud platform on EKS:
# 1. Infrastructure Setup (VPC, EKS, ECR) using ai-on-eks pattern
# 2. Dynamo Platform Setup (venv, clone repo, build images, deploy)
# 3. Blueprint Integration (scripts for inference graph deployment)
#
# The script follows the dynamo-cloud reference implementation
# but integrates with the ai-on-eks infrastructure patterns.
#---------------------------------------------------------------

set -euo pipefail

# Script directory and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLUEPRINT_DIR="$(cd "${SCRIPT_DIR}/../../blueprints/inference/nvidia-dynamo" && pwd)"
ENV_FILE="${BLUEPRINT_DIR}/dynamo_env.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Utility functions
info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

print_banner() {
    local title="$1"
    local width=80
    local line=$(printf '%*s' "$width" | tr ' ' '=')

    echo -e "\n${BLUE}${line}${NC}"
    echo -e "${BLUE}$(printf '%*s' $(( (width - ${#title}) / 2 )) '')${title}${NC}"
    echo -e "${BLUE}${line}${NC}\n"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

print_banner "NVIDIA DYNAMO ON EKS - INSTALLATION"

#---------------------------------------------------------------
# System Compatibility Check
#---------------------------------------------------------------

section "System Compatibility Check"

# Check if running on supported Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" = "ubuntu" ]; then
        case "$VERSION_ID" in
            "22.04"|"24.04")
                success "Detected supported Ubuntu version: $VERSION_ID"
                ;;
            *)
                error "Unsupported Ubuntu version: $VERSION_ID"
                error "NVIDIA Dynamo requires Ubuntu 22.04 or 24.04"
                error "Please use a supported Ubuntu version"
                exit 1
                ;;
        esac
    else
        error "Unsupported operating system: $ID"
        error "NVIDIA Dynamo requires Ubuntu 22.04 or 24.04"
        error "Please use a supported Ubuntu system"
        exit 1
    fi
else
    error "Cannot detect operating system"
    error "NVIDIA Dynamo requires Ubuntu 22.04 or 24.04"
    exit 1
fi

success "System compatibility check passed"

#---------------------------------------------------------------
# Phase 1: Infrastructure Setup (ai-on-eks pattern)
#---------------------------------------------------------------

section "Phase 1: Infrastructure Setup"
info "Setting up base infrastructure using ai-on-eks pattern..."

# Copy the base terraform into the local folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

# Copy dynamo-specific terraform files
info "Adding Dynamo-specific terraform configurations..."
cp ./terraform/dynamo-*.tf ./terraform/_LOCAL/
cp ./terraform/custom-*.tf ./terraform/_LOCAL/
cp ./terraform/blueprint.tfvars ./terraform/_LOCAL/

# Copy modified base files to overwrite the base versions
info "Applying Dynamo-specific modifications to base terraform files..."
cp ./terraform/variables.tf ./terraform/_LOCAL/
cp ./terraform/versions.tf ./terraform/_LOCAL/
cp ./terraform/addons.tf ./terraform/_LOCAL/
cp ./terraform/outputs.tf ./terraform/_LOCAL/

terraform init -upgrade

# Apply terraform
cd terraform/_LOCAL
info "Applying terraform infrastructure..."
source ./install.sh

if [ $? -eq 0 ]; then
    success "Infrastructure setup completed successfully"
else
    error "Infrastructure setup failed"
    exit 1
fi

# Get terraform outputs for later use
info "Extracting terraform outputs..."
export AWS_ACCOUNT_ID=$(terraform output -raw aws_account_id 2>/dev/null || aws sts get-caller-identity --query Account --output text)
export AWS_REGION=$(terraform output -raw aws_region 2>/dev/null || aws configure get region || echo "us-west-2")
export CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || echo "dynamo-on-eks")
export DYNAMO_REPO_VERSION=$(terraform output -raw dynamo_stack_version 2>/dev/null || echo "v0.3.1")

# Update kubeconfig
info "Updating kubeconfig for cluster access..."
aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}

# Return to script directory
cd "${SCRIPT_DIR}"

success "Phase 1 completed: Infrastructure is ready"

#---------------------------------------------------------------
# Phase 1.5: AWS Service-Linked Role Setup for Karpenter
#---------------------------------------------------------------

section "Phase 1.5: AWS Service-Linked Role Setup"

info "Checking and creating AWS Spot service-linked role for Karpenter..."

# Check if the Spot service-linked role exists
if aws iam get-role --role-name AWSServiceRoleForEC2Spot >/dev/null 2>&1; then
    info "AWS Spot service-linked role already exists"
else
    info "Creating AWS Spot service-linked role..."
    if aws iam create-service-linked-role --aws-service-name spot.amazonaws.com >/dev/null 2>&1; then
        success "AWS Spot service-linked role created successfully"
    else
        warn "Failed to create Spot service-linked role. This may be due to insufficient permissions."
        warn "Karpenter may have issues provisioning spot instances."
        warn "You can create it manually later with: aws iam create-service-linked-role --aws-service-name spot.amazonaws.com"
    fi
fi

# Verification loop to ensure service-linked role is actually set
info "Verifying AWS Spot service-linked role is properly configured..."
RETRY_COUNT=0
MAX_RETRIES=12
RETRY_DELAY=5

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if aws iam get-role --role-name AWSServiceRoleForEC2Spot >/dev/null 2>&1; then
        success "AWS Spot service-linked role verification successful"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            info "Service-linked role not yet available, retrying in ${RETRY_DELAY} seconds... (attempt ${RETRY_COUNT}/${MAX_RETRIES})"
            sleep $RETRY_DELAY
        else
            error "Service-linked role verification failed after ${MAX_RETRIES} attempts"
            error "This may cause issues with Karpenter spot instance provisioning"
            error "Please verify the role exists manually: aws iam get-role --role-name AWSServiceRoleForEC2Spot"
        fi
    fi
done

success "Phase 1.5 completed: AWS service-linked roles are ready"

#---------------------------------------------------------------
# Phase 2: Dynamo Platform Setup
#---------------------------------------------------------------

section "Phase 2: Dynamo Platform Setup"

# Create blueprint directory if it doesn't exist
mkdir -p "${BLUEPRINT_DIR}"

# Create environment file
info "Creating Dynamo environment configuration..."
cat > "${ENV_FILE}" << EOF
#!/bin/bash
# Dynamo Environment Configuration for ai-on-eks
export DYNAMO_REPO_VERSION="${DYNAMO_REPO_VERSION}"
export DYNAMO_FROM_SOURCE=false
export AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID}"
export AWS_REGION="${AWS_REGION}"
export CLUSTER_NAME="${CLUSTER_NAME}"
export NAMESPACE="dynamo-cloud"
export IMAGE_TAG="latest"

# ECR Repository names (must match terraform configuration)
export OPERATOR_ECR_REPOSITORY="dynamo-operator"
export API_STORE_ECR_REPOSITORY="dynamo-api-store"
export PIPELINES_ECR_REPOSITORY="dynamo-pipelines"
export BASE_ECR_REPOSITORY="dynamo-base"

# Docker registry settings
export DOCKER_SERVER="\${AWS_ACCOUNT_ID}.dkr.ecr.\${AWS_REGION}.amazonaws.com"
export CI_REGISTRY_IMAGE="\${AWS_ACCOUNT_ID}.dkr.ecr.\${AWS_REGION}.amazonaws.com"
export CI_COMMIT_SHA="\${IMAGE_TAG}"

# ECR credentials for Dynamo Cloud deployment
# These are used by the deploy script for buildkit to push pipeline images
export DOCKER_USERNAME="AWS"
export PIPELINES_DOCKER_USERNAME="AWS"
# Note: DOCKER_PASSWORD and PIPELINES_DOCKER_PASSWORD are set dynamically during deployment
# using: aws ecr get-login-password --region \${AWS_REGION}

# Completed scripts tracking
export COMPLETED_SCRIPTS=()
EOF

success "Environment configuration created at ${ENV_FILE}"

# Source the environment
source "${ENV_FILE}"

# Navigate to blueprint directory for venv and dynamo repo setup
cd "${BLUEPRINT_DIR}"

info "Setting up Python virtual environment in blueprint directory..."
# Install required packages
sudo apt-get update
sudo apt-get install -y python3-full python3-pip python3-venv

# Create and activate virtual environment
rm -rf dynamo_venv
python3 -m venv dynamo_venv
source dynamo_venv/bin/activate

# Upgrade pip and install dynamo
info "Installing ai-dynamo[all] package..."
pip install --upgrade pip
pip install -U ai-dynamo[all]==${DYNAMO_REPO_VERSION#v}
pip install tensorboardX

success "Dynamo Python package installed successfully"

# Clone dynamo repository for container build files
info "Cloning Dynamo repository for container builds..."
if [ -d "dynamo" ]; then
    info "Dynamo repository already exists. Updating to ${DYNAMO_REPO_VERSION}..."
    cd dynamo
    git fetch --tags
    git reset --hard

    # Try to checkout the tag first, if it fails, try the branch
    if git tag -l | grep -q "^${DYNAMO_REPO_VERSION}$"; then
        info "Found tag ${DYNAMO_REPO_VERSION}, checking out..."
        git checkout tags/${DYNAMO_REPO_VERSION}
    else
        info "Tag ${DYNAMO_REPO_VERSION} not found, trying branch release/${DYNAMO_REPO_VERSION}..."
        git checkout release/${DYNAMO_REPO_VERSION}
        git pull origin release/${DYNAMO_REPO_VERSION}
    fi
    cd ..
else
    info "Cloning Dynamo repository..."
    git clone https://github.com/ai-dynamo/dynamo.git
    cd dynamo
    git fetch --tags

    # Try to checkout the tag first, if it fails, try the branch
    if git tag -l | grep -q "^${DYNAMO_REPO_VERSION}$"; then
        info "Found tag ${DYNAMO_REPO_VERSION}, checking out..."
        git checkout tags/${DYNAMO_REPO_VERSION}
    else
        info "Tag ${DYNAMO_REPO_VERSION} not found, trying branch release/${DYNAMO_REPO_VERSION}..."
        git checkout release/${DYNAMO_REPO_VERSION}
    fi
    cd ..
fi

success "Dynamo repository cloned and configured"

# Verify Dynamo CLI installation
if ! command_exists dynamo; then
    error "Dynamo CLI not found in PATH. Installation may have failed."
    error "Make sure the virtual environment is activated."
    exit 1
fi

success "Dynamo CLI successfully installed and available"

#---------------------------------------------------------------
# Phase 3: Container Build and Push
#---------------------------------------------------------------

section "Phase 3: Container Build and Push"

# Login to ECR
info "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build operator and api-store images using Earthly
info "Building and pushing Dynamo operator and API store images..."

# Check if earthly is available
if ! command_exists earthly; then
    error "Earthly not found. Please install earthly first."
    exit 1
fi

# Build and push all platform components using the main Earthfile
info "Working in directory: $(pwd)"
info "Building and pushing operator and API store images..."
cd "${BLUEPRINT_DIR}"/dynamo
earthly --push +all-docker --DOCKER_SERVER=${DOCKER_SERVER} --IMAGE_TAG=${IMAGE_TAG}

if [ $? -eq 0 ]; then
    success "Operator and API store images built and pushed successfully"
else
    error "Operator and API store build failed"
    exit 1
fi

success "Platform container images built and pushed successfully"

cd "${BLUEPRINT_DIR}"

#---------------------------------------------------------------
# Phase 3.5: Image Build Engine Configuration (Optional)
#---------------------------------------------------------------

section "Phase 3.5: Image Build Engine Configuration"

# Ask user about image build engine preference
info "Dynamo supports different image build engines:"
info "  - kaniko (default): Rootless builds, enhanced security, works well with IRSA"
info "  - buildkit: Fast, efficient, good compatibility"
info ""
echo -n "Would you like to use buildkit instead of kaniko? (y/N): "
read -r USE_BUILDKIT

if [[ "${USE_BUILDKIT,,}" =~ ^(y|yes)$ ]]; then
    info "Keeping buildkit build engine..."
else
    info "Configuring Dynamo to use kaniko build engine..."

    # Path to the dynamo-platform-values.yaml file
    VALUES_FILE="${BLUEPRINT_DIR}/dynamo/deploy/cloud/helm/dynamo-platform-values.yaml"

    # Check if the file exists
    if [ -f "${VALUES_FILE}" ]; then
        # Update imageBuildEngine from buildkit to kaniko
        if grep -q "imageBuildEngine: buildkit" "${VALUES_FILE}"; then
            sed -i 's/imageBuildEngine: buildkit/imageBuildEngine: kaniko/' "${VALUES_FILE}"
            success "Updated imageBuildEngine to kaniko in ${VALUES_FILE}"
        else
            warn "imageBuildEngine: buildkit not found in expected format"
            warn "Please manually update the imageBuildEngine setting in:"
            warn "${VALUES_FILE}"
        fi
    else
        error "Values file not found: ${VALUES_FILE}"
        error "Cannot update image build engine configuration"
        exit 1
    fi
fi

success "Phase 3.5 completed: Image build engine configured"

#---------------------------------------------------------------
# Phase 3.6: IRSA Configuration for ECR Access
#---------------------------------------------------------------

section "Phase 3.6: IRSA Configuration for ECR Access"

info "Configuring IRSA for ECR access to eliminate credential rotation..."

# Get the IRSA role ARN from terraform output
cd "${SCRIPT_DIR}/terraform/_LOCAL"
ECR_ROLE_ARN=$(terraform output -raw dynamo_ecr_role_arn 2>/dev/null || echo "")
cd "${BLUEPRINT_DIR}"

if [ -n "$ECR_ROLE_ARN" ]; then
    info "Found IRSA role ARN: $ECR_ROLE_ARN"

    # Path to the dynamo-platform-values.yaml file
    VALUES_FILE="${BLUEPRINT_DIR}/dynamo/deploy/cloud/helm/dynamo-platform-values.yaml"

    if [ -f "${VALUES_FILE}" ]; then
        info "Configuring dynamo-platform-values.yaml for IRSA..."

        # 1. Disable kubernetes secret usage
        sed -i 's/useKubernetesSecret: true/useKubernetesSecret: false/' "${VALUES_FILE}"

        success "Disabled kubernetes secret usage in platform values"
        info "ECR access will be configured via IRSA after deployment"

        # Update environment to skip ECR credential setup
        export SKIP_ECR_CREDENTIALS=true

    else
        error "Values file not found: ${VALUES_FILE}"
        exit 1
    fi
else
    warn "IRSA role ARN not found in terraform outputs"
    warn "Falling back to ECR credential rotation approach"
    export SKIP_ECR_CREDENTIALS=false
fi

success "Phase 3.6 completed: IRSA configuration applied"

#---------------------------------------------------------------
# Phase 4: Dynamo Platform Deployment
#---------------------------------------------------------------

section "Phase 4: Dynamo Platform Deployment"

# Create namespace
info "Creating Dynamo namespace..."
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Deploy Dynamo platform using the official deploy script approach
info "Deploying Dynamo platform..."
cd "${BLUEPRINT_DIR}"/dynamo

# Use the deploy script from the dynamo repository
if [ -f "deploy/helm/deploy.sh" ]; then
    info "Using official Dynamo deploy script..."
    cd deploy/cloud/helm

    # Set environment variables for the deploy script
    export NAMESPACE=${NAMESPACE}
    export DOCKER_SERVER=${DOCKER_SERVER}
    export PIPELINES_DOCKER_SERVER=${DOCKER_SERVER}
    export OPERATOR_IMAGE=${DOCKER_SERVER}/${OPERATOR_ECR_REPOSITORY}:${IMAGE_TAG}
    export API_STORE_IMAGE=${DOCKER_SERVER}/${API_STORE_ECR_REPOSITORY}:${IMAGE_TAG}

    # Set ECR credentials for buildkit to push pipeline images (if not using IRSA)
    if [ "${SKIP_ECR_CREDENTIALS:-false}" = "true" ]; then
        info "Using IRSA for ECR access - skipping credential setup"
        info "Service accounts will authenticate to ECR using IAM roles"

        # Still need to set these for the deploy script, but they won't be used by the pods
        export DOCKER_USERNAME="AWS"
        export DOCKER_PASSWORD="unused-with-irsa"
        export PIPELINES_DOCKER_USERNAME="AWS"
        export PIPELINES_DOCKER_PASSWORD="unused-with-irsa"
    else
        # According to dynamo/docs/guides/dynamo_deploy/dynamo_cloud.md:
        # - DOCKER_USERNAME/DOCKER_PASSWORD: for pulling platform component images
        # - PIPELINES_DOCKER_USERNAME/PIPELINES_DOCKER_PASSWORD: for buildkit to push pipeline images
        info "Getting ECR credentials for buildkit..."

        ECR_PASSWORD=$(aws ecr get-login-password --region ${AWS_REGION})
        if [ -z "$ECR_PASSWORD" ]; then
            error "Failed to get ECR login password. Check AWS credentials and permissions."
            exit 1
        fi

        # Set credentials for platform component image pulls
        export DOCKER_USERNAME="AWS"
        export DOCKER_PASSWORD="$ECR_PASSWORD"

        # Set credentials for buildkit to push pipeline images to ECR
        export PIPELINES_DOCKER_USERNAME="AWS"
        export PIPELINES_DOCKER_PASSWORD="$ECR_PASSWORD"

        info "ECR credentials configured:"
        info "- Platform component pulls: DOCKER_USERNAME/DOCKER_PASSWORD"
        info "- Pipeline image pushes: PIPELINES_DOCKER_USERNAME/PIPELINES_DOCKER_PASSWORD"
    fi

    # Run the deploy script
    ./deploy.sh --crds

    if [ $? -eq 0 ]; then
        success "Dynamo platform deployed successfully"
    else
        error "Dynamo platform deployment failed"
        exit 1
    fi
else
    error "Deploy script not found in dynamo repository"
    exit 1
fi

cd "${BLUEPRINT_DIR}"

success "Phase 4 completed: Dynamo platform is deployed"

#---------------------------------------------------------------
# Phase 4.5: IRSA Configuration and ECR Access Setup
#---------------------------------------------------------------

section "Phase 4.5: IRSA Configuration and ECR Access Setup"

# Apply IRSA annotations after deployment to ensure ECR access works
if [ "${SKIP_ECR_CREDENTIALS:-false}" = "true" ] && [ -n "${ECR_ROLE_ARN:-}" ]; then
    info "Configuring IRSA for ECR access after deployment..."

    # Wait for Dynamo service accounts to be created
    info "Waiting for Dynamo service accounts to be ready..."
    for i in {1..60}; do
        SA_COUNT=$(kubectl get serviceaccount -n ${NAMESPACE} 2>/dev/null | grep "dynamo-cloud-dynamo-" | wc -l)
        if [ "$SA_COUNT" -ge 3 ]; then
            success "Found $SA_COUNT Dynamo service accounts"
            break
        fi
        if [ $i -eq 60 ]; then
            warn "Service accounts not ready after 10 minutes"
            break
        else
            info "Waiting for service accounts... ($i/60)"
            sleep 10
        fi
    done

    # Apply IRSA annotations to all Dynamo service accounts
    info "Applying IRSA annotations for ECR access..."

    # Controller manager service account
    if kubectl get serviceaccount dynamo-cloud-dynamo-operator-controller-manager -n ${NAMESPACE} >/dev/null 2>&1; then
        kubectl annotate serviceaccount dynamo-cloud-dynamo-operator-controller-manager -n ${NAMESPACE} eks.amazonaws.com/role-arn=${ECR_ROLE_ARN} --overwrite
        success "IRSA annotation applied to controller-manager service account"
    else
        warn "Controller manager service account not found"
    fi

    # Image builder service account
    if kubectl get serviceaccount dynamo-cloud-dynamo-operator-image-builder -n ${NAMESPACE} >/dev/null 2>&1; then
        kubectl annotate serviceaccount dynamo-cloud-dynamo-operator-image-builder -n ${NAMESPACE} eks.amazonaws.com/role-arn=${ECR_ROLE_ARN} --overwrite
        success "IRSA annotation applied to image-builder service account"
    else
        warn "Image builder service account not found"
    fi

    # Component service account (if it exists)
    if kubectl get serviceaccount dynamo-cloud-dynamo-operator-component -n ${NAMESPACE} >/dev/null 2>&1; then
        kubectl annotate serviceaccount dynamo-cloud-dynamo-operator-component -n ${NAMESPACE} eks.amazonaws.com/role-arn=${ECR_ROLE_ARN} --overwrite
        success "IRSA annotation applied to component service account"
    else
        info "Component service account not found (this is normal)"
    fi

    # API Store service account
    if kubectl get serviceaccount dynamo-cloud-dynamo-api-store -n ${NAMESPACE} >/dev/null 2>&1; then
        kubectl annotate serviceaccount dynamo-cloud-dynamo-api-store -n ${NAMESPACE} eks.amazonaws.com/role-arn=${ECR_ROLE_ARN} --overwrite
        success "IRSA annotation applied to api-store service account"
    else
        warn "API Store service account not found"
    fi

    # Remove ECR credential secrets (not needed with IRSA)
    info "Cleaning up ECR credential secrets (not needed with IRSA)..."
    kubectl delete secret docker-imagepullsecret -n ${NAMESPACE} --ignore-not-found=true
    kubectl delete secret dynamo-regcred -n ${NAMESPACE} --ignore-not-found=true
    success "ECR credential secrets removed"

    # Wait for all pods to be ready
    info "Waiting for all Dynamo pods to be ready..."
    for i in {1..60}; do
        READY_PODS=$(kubectl get pods -n ${NAMESPACE} --no-headers 2>/dev/null | grep -E "(Running|Completed)" | wc -l)
        TOTAL_PODS=$(kubectl get pods -n ${NAMESPACE} --no-headers 2>/dev/null | wc -l)

        if [ "$READY_PODS" -eq "$TOTAL_PODS" ] && [ "$TOTAL_PODS" -gt 0 ]; then
            success "All $TOTAL_PODS Dynamo pods are ready"
            break
        fi
        if [ $i -eq 60 ]; then
            warn "Not all pods ready after 10 minutes ($READY_PODS/$TOTAL_PODS ready)"
            warn "Check pod status with: kubectl get pods -n ${NAMESPACE}"
        else
            info "Waiting for pods to be ready... ($READY_PODS/$TOTAL_PODS ready) ($i/60)"
            sleep 10
        fi
    done

    success "IRSA configuration completed - ECR access is now working with IAM roles"
else
    info "Skipping IRSA configuration (using ECR credential rotation)"
fi

success "Phase 4.5 completed: IRSA and ECR access configured"

#---------------------------------------------------------------
# Phase 4.6: Karpenter Verification
#---------------------------------------------------------------

section "Phase 4.6: Karpenter Verification"

info "Verifying Karpenter NodePools are ready..."

# Wait for NodePools to be created
info "Waiting for Karpenter NodePools to be ready..."
for i in {1..30}; do
    NODEPOOL_COUNT=$(kubectl get nodepool --no-headers 2>/dev/null | wc -l)
    if [ "$NODEPOOL_COUNT" -ge 5 ]; then
        success "Found $NODEPOOL_COUNT NodePools - Karpenter is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        warn "NodePools not ready after 5 minutes. Karpenter may need manual intervention."
        warn "Check with: kubectl get nodepool -o wide"
    else
        info "Waiting for NodePools... ($i/30)"
        sleep 10
    fi
done

# Display NodePool status
info "Current NodePool status:"
kubectl get nodepool -o wide 2>/dev/null || warn "Unable to get NodePool status"

# Check for any failed NodeClaims
FAILED_NODECLAIMS=$(kubectl get nodeclaims --no-headers 2>/dev/null | grep -v "True\|Unknown" | wc -l)
if [ "$FAILED_NODECLAIMS" -gt 0 ]; then
    warn "Found $FAILED_NODECLAIMS failed NodeClaims. Check with: kubectl get nodeclaims -o wide"
else
    info "No failed NodeClaims detected"
fi

success "Phase 4.5 completed: Karpenter verification complete"

#---------------------------------------------------------------
# Phase 5: Blueprint Scripts Setup
#---------------------------------------------------------------

section "Phase 5: Blueprint Scripts Setup"

info "Creating blueprint scripts for inference graph deployment..."

# Return to script directory to access the scripts we'll create
cd "${SCRIPT_DIR}"

# The scripts will be created by the next part of the installation
# This completes the infrastructure and platform setup

success "Installation completed successfully!"

echo "================================================"
echo "Next steps:"
echo "1. Navigate to blueprints/inference/nvidia-dynamo"
echo "2. Activate the virtual environment: source dynamo_venv/bin/activate"
echo "3. Use the deploy.sh script to deploy inference graphs"
echo "4. Use the test.sh script to test deployments"
echo "================================================"
echo "Environment file: ${ENV_FILE}"
echo "Virtual environment: ${BLUEPRINT_DIR}/dynamo_venv"
echo "Dynamo repository: ${BLUEPRINT_DIR}/dynamo"
echo "================================================"
echo "ECR Authentication:"
if [ "${SKIP_ECR_CREDENTIALS:-false}" = "true" ]; then
    echo "- Using IRSA (IAM Roles for Service Accounts) for ECR access"
    echo "- No credential rotation needed"
    echo "- Service accounts automatically authenticate to ECR"
    echo "- Image build engine: kaniko (default, works best with IRSA)"
else
    echo "- Using ECR credential rotation (legacy mode)"
    echo "- Automatic refresh: CronJob runs every 6 hours"
    echo "- Monitor status: kubectl get cronjob ecr-token-refresh -n ${NAMESPACE}"
fi
echo "================================================"
echo "Karpenter Status:"
echo "- Check NodePools: kubectl get nodepool -o wide"
echo "- Check NodeClaims: kubectl get nodeclaims -o wide"
echo "- Check nodes: kubectl get nodes"
echo "================================================"
