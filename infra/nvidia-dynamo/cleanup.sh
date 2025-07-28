#!/bin/bash

#---------------------------------------------------------------
# NVIDIA Dynamo on EKS - Cleanup Script
#
# This script cleans up all resources created by the Dynamo
# installation, including:
# 1. Dynamo platform components
# 2. Kubernetes resources
# 3. ECR repositories and images
# 4. Terraform infrastructure
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

print_banner "NVIDIA DYNAMO ON EKS - CLEANUP"

# Load environment if available
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
    info "Loaded environment configuration from ${ENV_FILE}"
else
    warn "Environment file not found, using defaults"
    export AWS_REGION="${AWS_REGION:-us-west-2}"
    export CLUSTER_NAME="${CLUSTER_NAME:-dynamo-on-eks}"
    export NAMESPACE="${NAMESPACE:-dynamo-cloud}"
fi

#---------------------------------------------------------------
# Phase 1: Dynamo Platform Cleanup
#---------------------------------------------------------------

section "Phase 1: Dynamo Platform Cleanup"

# Update kubeconfig if cluster exists
info "Checking cluster connectivity..."
if aws eks describe-cluster --name ${CLUSTER_NAME} --region ${AWS_REGION} >/dev/null 2>&1; then
    info "Updating kubeconfig for cluster: ${CLUSTER_NAME}"
    aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}

    # Remove Dynamo deployments using dynamo CLI first
    info "Removing Dynamo deployments using dynamo CLI..."

    # Set up environment for dynamo CLI
    BLUEPRINT_DIR="${SCRIPT_DIR}/../../blueprints/inference/nvidia-dynamo"
    DYNAMO_VENV="${BLUEPRINT_DIR}/dynamo_venv"
    echo "DYNAMO_VENV: ${DYNAMO_VENV}"

    if [ -d "${DYNAMO_VENV}" ] && [ -f "${DYNAMO_VENV}/bin/activate" ]; then
        info "Activating Dynamo virtual environment..."
        source "${DYNAMO_VENV}/bin/activate"

        # Set up DYNAMO_CLOUD endpoint for API access
        info "Setting up Dynamo Cloud endpoint..."
        DYNAMO_STORE_SERVICE=""
        if kubectl get svc -n "${NAMESPACE}" dynamo-store >/dev/null 2>&1; then
            DYNAMO_STORE_SERVICE="dynamo-store"
        elif kubectl get svc -n "${NAMESPACE}" dynamo-cloud-dynamo-api-store >/dev/null 2>&1; then
            DYNAMO_STORE_SERVICE="dynamo-cloud-dynamo-api-store"
        fi

        if [ -n "$DYNAMO_STORE_SERVICE" ]; then
            info "Found Dynamo store service: ${DYNAMO_STORE_SERVICE}"

            # Start port-forward in background
            info "Starting port-forward for Dynamo API access..."
            kubectl port-forward svc/${DYNAMO_STORE_SERVICE} 8080:80 -n ${NAMESPACE} >/dev/null 2>&1 &
            PORT_FORWARD_PID=$!
            export DYNAMO_CLOUD="http://localhost:8080"

            # Wait for port-forward to be ready
            sleep 5

            # List and delete all deployments
            info "Listing Dynamo deployments..."
            if command -v dynamo >/dev/null 2>&1; then
                # Get the raw output from dynamo deployment list
                DEPLOYMENT_OUTPUT=$(dynamo deployment list --endpoint "${DYNAMO_CLOUD}" 2>/dev/null || echo "")

                if [ -n "${DEPLOYMENT_OUTPUT}" ]; then
                    info "Raw deployment list output:"
                    echo "${DEPLOYMENT_OUTPUT}"

                    # Extract deployment names using regex to handle pretty-printed box format
                    # Look for lines with "Name: <deployment-name>" pattern
                    DEPLOYMENTS=$(echo "${DEPLOYMENT_OUTPUT}" | \
                        grep -E "^\s*│\s*Name:\s+" | \
                        sed -E 's/^\s*│\s*Name:\s+([a-zA-Z0-9][a-zA-Z0-9_-]*)\s*.*$/\1/' | \
                        grep -E "^[a-zA-Z0-9][a-zA-Z0-9_-]*$" || echo "")

                    if [ -n "${DEPLOYMENTS}" ]; then
                        info "Found deployments to delete:"
                        echo "${DEPLOYMENTS}"

                        for deployment in ${DEPLOYMENTS}; do
                            info "Deleting Dynamo deployment: ${deployment}"
                            dynamo deployment delete "${deployment}" --endpoint "${DYNAMO_CLOUD}" || warn "Failed to delete deployment: ${deployment}"
                        done

                        # Wait for deployments to be cleaned up
                        info "Waiting for deployments to be cleaned up..."
                        sleep 10
                    else
                        info "No valid deployment names found in output"
                    fi
                else
                    info "No Dynamo deployments found or command failed"
                fi
            else
                warn "Dynamo CLI not found in virtual environment"
            fi

            # Kill port-forward
            if [ -n "${PORT_FORWARD_PID}" ]; then
                kill ${PORT_FORWARD_PID} 2>/dev/null || true
            fi
        else
            warn "No Dynamo store service found, skipping deployment deletion"
        fi

        # Deactivate virtual environment
        deactivate 2>/dev/null || true
    else
        warn "Dynamo virtual environment not found, skipping deployment deletion"
    fi

    # Now patch finalizers on Dynamo custom resources to prevent hanging
    info "Patching finalizers on Dynamo custom resources..."

    # Get all Dynamo custom resources in the namespace
    DYNAMO_CRS=(
        "dynamocomponentdeployments"
        "dynamocomponents"
        "dynamographdeployments"
    )

    for cr_type in "${DYNAMO_CRS[@]}"; do
        info "Processing ${cr_type}..."
        # Get all resources of this type in the namespace
        CR_NAMES=$(kubectl get ${cr_type} -n ${NAMESPACE} --no-headers -o custom-columns=":metadata.name" 2>/dev/null || echo "")

        if [ -n "${CR_NAMES}" ]; then
            for cr_name in ${CR_NAMES}; do
                info "Removing finalizers from ${cr_type}/${cr_name}..."
                if kubectl get ${cr_type} ${cr_name} -n ${NAMESPACE} >/dev/null 2>&1; then
                    kubectl patch ${cr_type} ${cr_name} -n ${NAMESPACE} --type='merge' -p='{"metadata":{"finalizers":null}}' || warn "Failed to patch finalizers on ${cr_type}/${cr_name}"
                else
                    info "Resource ${cr_type}/${cr_name} not found, skipping"
                fi
            done
        else
            info "No ${cr_type} found in namespace ${NAMESPACE}"
        fi
    done

    # Wait a moment for finalizer patches to take effect
    info "Waiting for finalizer patches to take effect..."
    sleep 5

    # Delete the namespace first to avoid pod disruption budget issues
    info "Deleting namespace ${NAMESPACE}..."
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true --timeout=300s || warn "Namespace deletion timed out, continuing with cleanup"

    # Wait for namespace deletion to complete
    info "Waiting for namespace deletion to complete..."
    for i in {1..30}; do
        if ! kubectl get namespace ${NAMESPACE} >/dev/null 2>&1; then
            success "Namespace ${NAMESPACE} deleted successfully"
            break
        fi
        if [ $i -eq 30 ]; then
            warn "Namespace deletion taking longer than expected, continuing with cleanup"
        else
            info "Waiting for namespace deletion... ($i/30)"
            sleep 10
        fi
    done

    # Now drain and delete Karpenter-managed nodes
    info "Draining and deleting Karpenter-managed nodes..."

    # Get all nodes managed by Karpenter
    KARPENTER_NODES=$(kubectl get nodes -l karpenter.sh/provisioner-name --no-headers -o custom-columns=":metadata.name" 2>/dev/null || echo "")
    if [ -z "${KARPENTER_NODES}" ]; then
        # Try alternative label format for newer Karpenter versions
        KARPENTER_NODES=$(kubectl get nodes -l karpenter.sh/nodepool --no-headers -o custom-columns=":metadata.name" 2>/dev/null || echo "")
    fi

    if [ -n "${KARPENTER_NODES}" ]; then
        for node in ${KARPENTER_NODES}; do
            info "Draining node: ${node}"
            kubectl drain ${node} --ignore-daemonsets --delete-emptydir-data --force --timeout=300s || warn "Failed to drain node: ${node}"

            info "Deleting node: ${node}"
            kubectl delete node ${node} --timeout=60s || warn "Failed to delete node: ${node}"
        done

        # Wait for nodes to be fully terminated
        info "Waiting for nodes to be terminated..."
        sleep 30
    else
        info "No Karpenter-managed nodes found"
    fi

    # Also force delete any stuck NodeClaims
    info "Cleaning up Karpenter NodeClaims..."
    NODECLAIMS=$(kubectl get nodeclaims --no-headers -o custom-columns=":metadata.name" 2>/dev/null || echo "")
    if [ -n "${NODECLAIMS}" ]; then
        for nodeclaim in ${NODECLAIMS}; do
            info "Removing finalizers from NodeClaim: ${nodeclaim}"
            kubectl patch nodeclaim ${nodeclaim} --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || true

            info "Deleting NodeClaim: ${nodeclaim}"
            kubectl delete nodeclaim ${nodeclaim} --timeout=60s || warn "Failed to delete NodeClaim: ${nodeclaim}"
        done
    else
        info "No NodeClaims found"
    fi

    # Remove any Dynamo CRDs
    info "Removing Dynamo Custom Resource Definitions..."
    kubectl delete crd -l app.kubernetes.io/name=dynamo --ignore-not-found=true

    # Remove any remaining Dynamo resources
    info "Cleaning up any remaining Dynamo resources..."
    kubectl delete all -l app.kubernetes.io/part-of=dynamo --all-namespaces --ignore-not-found=true

    success "Dynamo platform cleanup completed"
else
    warn "Cluster ${CLUSTER_NAME} not found or not accessible, skipping Kubernetes cleanup"
fi

#---------------------------------------------------------------
# Phase 2: ECR Repository Cleanup
#---------------------------------------------------------------

section "Phase 2: ECR Repository Cleanup"

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")

if [ -n "${AWS_ACCOUNT_ID}" ]; then
    info "Cleaning up ECR repositories..."

    # List of ECR repositories to clean up
    ECR_REPOS=(
        "dynamo-operator"
        "dynamo-api-store"
        "dynamo-pipelines"
        "dynamo-base"
    )

    for repo in "${ECR_REPOS[@]}"; do
        info "Checking ECR repository: ${repo}"
        if aws ecr describe-repositories --repository-names ${repo} --region ${AWS_REGION} >/dev/null 2>&1; then
            info "Deleting ECR repository: ${repo}"
            aws ecr delete-repository --repository-name ${repo} --region ${AWS_REGION} --force || warn "Failed to delete repository: ${repo}"
        else
            info "Repository ${repo} not found, skipping"
        fi
    done

    success "ECR repository cleanup completed"
else
    warn "Could not determine AWS account ID, skipping ECR cleanup"
fi

#---------------------------------------------------------------
# Phase 3: Terraform Infrastructure Cleanup
#---------------------------------------------------------------

section "Phase 3: Terraform Infrastructure Cleanup"

# Navigate to terraform directory
TERRAFORM_DIR="${SCRIPT_DIR}/terraform/_LOCAL"

if [ -d "${TERRAFORM_DIR}" ]; then
    info "Found terraform directory: ${TERRAFORM_DIR}"
    cd "${TERRAFORM_DIR}"

    # Initialize terraform
    info "Initializing Terraform..."
    terraform init

    # Destroy infrastructure
    info "Destroying Terraform infrastructure..."
    terraform destroy -auto-approve -var-file=../blueprint.tfvars

    if [ $? -eq 0 ]; then
        success "Terraform infrastructure destroyed successfully"

        # Clean up terraform directory
        info "Cleaning up terraform working directory..."
        cd "${SCRIPT_DIR}"
        rm -rf "${TERRAFORM_DIR}"
        success "Terraform working directory cleaned up"
    else
        error "Terraform destroy failed"
        warn "Manual cleanup may be required"
    fi
else
    warn "Terraform directory not found, skipping infrastructure cleanup"
fi

#---------------------------------------------------------------
# Phase 4: Blueprint Cleanup
#---------------------------------------------------------------

section "Phase 4: Blueprint Cleanup"

if [ -d "${BLUEPRINT_DIR}" ]; then
    info "Cleaning up blueprint directory..."

    # Remove virtual environment
    if [ -d "${BLUEPRINT_DIR}/dynamo_venv" ]; then
        info "Removing Python virtual environment..."
        rm -rf "${BLUEPRINT_DIR}/dynamo_venv"
    fi

    # Remove dynamo repository
    if [ -d "${BLUEPRINT_DIR}/dynamo" ]; then
        info "Removing Dynamo repository clone..."
        rm -rf "${BLUEPRINT_DIR}/dynamo"
    fi

    # Remove environment file
    if [ -f "${ENV_FILE}" ]; then
        info "Removing environment configuration..."
        rm -f "${ENV_FILE}"
    fi

    success "Blueprint cleanup completed"
else
    warn "Blueprint directory not found, skipping blueprint cleanup"
fi

#---------------------------------------------------------------
# Phase 5: Additional AWS Resource Cleanup
#---------------------------------------------------------------

section "Phase 5: Additional AWS Resource Cleanup"

info "Cleaning up additional AWS resources..."

# Clean up any remaining EFS file systems
info "Checking for EFS file systems..."
EFS_FILESYSTEMS=$(aws efs describe-file-systems --region ${AWS_REGION} --query "FileSystems[?contains(Tags[?Key=='Name'].Value, 'dynamo')].FileSystemId" --output text 2>/dev/null || echo "")

if [ -n "${EFS_FILESYSTEMS}" ]; then
    for fs_id in ${EFS_FILESYSTEMS}; do
        info "Found EFS file system: ${fs_id}"

        # Delete mount targets first
        MOUNT_TARGETS=$(aws efs describe-mount-targets --file-system-id ${fs_id} --region ${AWS_REGION} --query "MountTargets[].MountTargetId" --output text 2>/dev/null || echo "")

        for mt_id in ${MOUNT_TARGETS}; do
            info "Deleting mount target: ${mt_id}"
            aws efs delete-mount-target --mount-target-id ${mt_id} --region ${AWS_REGION} || warn "Failed to delete mount target: ${mt_id}"
        done

        # Wait for mount targets to be deleted
        info "Waiting for mount targets to be deleted..."
        sleep 30

        # Delete file system
        info "Deleting EFS file system: ${fs_id}"
        aws efs delete-file-system --file-system-id ${fs_id} --region ${AWS_REGION} || warn "Failed to delete EFS file system: ${fs_id}"
    done
fi

# Clean up CloudWatch log groups
info "Cleaning up CloudWatch log groups..."
LOG_GROUPS=$(aws logs describe-log-groups --region ${AWS_REGION} --log-group-name-prefix "/aws/eks/${CLUSTER_NAME}" --query "logGroups[].logGroupName" --output text 2>/dev/null || echo "")

for log_group in ${LOG_GROUPS}; do
    info "Deleting CloudWatch log group: ${log_group}"
    aws logs delete-log-group --log-group-name ${log_group} --region ${AWS_REGION} || warn "Failed to delete log group: ${log_group}"
done

success "Additional AWS resource cleanup completed"

#---------------------------------------------------------------
# Summary
#---------------------------------------------------------------

section "Cleanup Summary"

success "NVIDIA Dynamo on EKS cleanup completed!"

echo ""
echo "Cleaned up resources:"
echo "  ✓ Dynamo platform components"
echo "  ✓ Kubernetes namespace and resources"
echo "  ✓ ECR repositories and images"
echo "  ✓ Terraform infrastructure"
echo "  ✓ Blueprint directory and files"
echo "  ✓ Additional AWS resources"
echo ""
echo "Note: Some resources may take time to be fully deleted."
echo "Check the AWS console to verify all resources have been removed."
echo ""
echo "If you encounter any issues, you may need to manually clean up:"
echo "  - VPC and networking components"
echo "  - IAM roles and policies"
echo "  - Security groups"
echo "  - Load balancers"
