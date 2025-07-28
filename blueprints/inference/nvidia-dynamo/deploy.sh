#!/bin/bash

#---------------------------------------------------------------
# NVIDIA Dynamo Inference Graph Deployment
#
# This script builds and deploys inference graphs using the Dynamo
# Cloud platform. It combines the functionality of the 5a and 6a
# scripts from the dynamo-cloud reference implementation and includes
# automatic Service and ServiceMonitor creation for monitoring.
#
# Phases:
#   1. Build Inference Graph (5a equivalent)
#   2. Deploy Inference Graph (6a equivalent)
#   3. Create Service and ServiceMonitor for monitoring
#   4. Service Exposure and Testing
#
# Usage:
#   ./deploy.sh [example_type] [llm_architecture]
#
# Examples:
#   ./deploy.sh llm agg          # Deploy LLM with aggregated architecture
#   ./deploy.sh llm disagg       # Deploy LLM with disaggregated architecture
#   ./deploy.sh hello-world      # Deploy hello-world example
#   ./deploy.sh                  # Interactive selection
#
# LLM Architectures:
#   agg, agg_router, disagg, disagg_router, multinode-405b, multinode_agg_r1, mutinode_disagg_r1
#---------------------------------------------------------------

set -euo pipefail

# Script directory and environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/dynamo_env.sh"

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

print_banner "DYNAMO INFERENCE GRAPH DEPLOYMENT"

# Load environment configuration
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
    info "Loaded environment configuration from ${ENV_FILE}"
else
    error "Environment file not found: ${ENV_FILE}"
    error "Please run the infrastructure installation first: infra/nvidia-dynamo/install.sh"
    exit 1
fi

# Activate virtual environment
if [ -d "${SCRIPT_DIR}/dynamo_venv" ]; then
    info "Activating Python virtual environment..."
    source "${SCRIPT_DIR}/dynamo_venv/bin/activate"
    success "Virtual environment activated"
else
    error "Virtual environment not found at ${SCRIPT_DIR}/dynamo_venv"
    error "Please run the infrastructure installation first: infra/nvidia-dynamo/install.sh"
    exit 1
fi

# Verify Dynamo CLI
if ! command_exists dynamo; then
    error "Dynamo CLI not found in PATH"
    error "Make sure the virtual environment is activated and Dynamo is installed"
    exit 1
fi

# Check if dynamo repository exists
if [ ! -d "${SCRIPT_DIR}/dynamo" ]; then
    error "Dynamo repository not found at ${SCRIPT_DIR}/dynamo"
    error "Please run the infrastructure installation first: infra/nvidia-dynamo/install.sh"
    exit 1
fi

# Set up DYNAMO_CLOUD endpoint (6a pattern)
section "Configuring Dynamo Cloud Endpoint"
if [ -z "${DYNAMO_CLOUD:-}" ]; then
    warn "DYNAMO_CLOUD environment variable is not set."
    info "Setting DYNAMO_CLOUD to http://localhost:8080 for local testing."
    export DYNAMO_CLOUD="http://localhost:8080"

    # Check for available Dynamo store services
    info "Checking for available Dynamo store services..."
    DYNAMO_STORE_SERVICE=""
    if kubectl get svc -n "${NAMESPACE}" dynamo-cloud-dynamo-api-store >/dev/null 2>&1; then
        DYNAMO_STORE_SERVICE="dynamo-cloud-dynamo-api-store"
    elif kubectl get svc -n "${NAMESPACE}" dynamo-store >/dev/null 2>&1; then
        DYNAMO_STORE_SERVICE="dynamo-store"
    fi

    if [ -n "$DYNAMO_STORE_SERVICE" ]; then
        success "Found Dynamo store service: ${DYNAMO_STORE_SERVICE}"
        info ""
        warn "MANUAL ACTION REQUIRED:"
        warn "Please set up port-forwarding for the Dynamo API store service."
        warn ""
        warn "In a separate terminal, run the following command:"
        echo "  kubectl port-forward svc/${DYNAMO_STORE_SERVICE} 8080:80 -n ${NAMESPACE}"
        warn ""
        warn "Keep that terminal open during the entire deployment process."
        warn ""
        info "Once port-forwarding is active, press ENTER to continue..."
        read -r

        # Test the connection
        info "Testing connection to Dynamo Cloud endpoint..."
        if curl -s --connect-timeout 5 "${DYNAMO_CLOUD}/health" >/dev/null 2>&1 || \
           curl -s --connect-timeout 5 "${DYNAMO_CLOUD}/" >/dev/null 2>&1; then
            success "Successfully connected to Dynamo Cloud endpoint"
        else
            error "Failed to connect to Dynamo Cloud endpoint at ${DYNAMO_CLOUD}"
            error "Please ensure port-forwarding is active and try again."
            error ""
            error "Debug steps:"
            error "1. Verify port-forwarding is running:"
            error "   kubectl port-forward svc/${DYNAMO_STORE_SERVICE} 8080:80 -n ${NAMESPACE}"
            error "2. Test the connection manually:"
            error "   curl ${DYNAMO_CLOUD}/"
            error "3. Check service status:"
            error "   kubectl get svc ${DYNAMO_STORE_SERVICE} -n ${NAMESPACE}"
            exit 1
        fi
    else
        error "Could not find any Dynamo API store service in namespace ${NAMESPACE}"
        error "Available services in namespace ${NAMESPACE}:"
        kubectl get svc -n "${NAMESPACE}" 2>/dev/null || echo "  No services found"
        error ""
        error "Please ensure the Dynamo infrastructure is properly installed."
        error "Run: infra/nvidia-dynamo/install.sh"
        exit 1
    fi
else
    success "DYNAMO_CLOUD is already set to: ${DYNAMO_CLOUD}"

    # Test the connection if it's a localhost endpoint
    if [[ "${DYNAMO_CLOUD}" == *"localhost"* ]]; then
        info "Testing connection to Dynamo Cloud endpoint..."
        if curl -s --connect-timeout 5 "${DYNAMO_CLOUD}/health" >/dev/null 2>&1 || \
           curl -s --connect-timeout 5 "${DYNAMO_CLOUD}/" >/dev/null 2>&1; then
            success "Successfully connected to Dynamo Cloud endpoint"
        else
            warn "Could not connect to Dynamo Cloud endpoint at ${DYNAMO_CLOUD}"
            warn "If using localhost, ensure port-forwarding is active."
        fi
    fi
fi

#---------------------------------------------------------------
# Example Selection (5a pattern)
#---------------------------------------------------------------

section "Example Selection"

# Select example type
EXAMPLE_TYPE=""
LLM_GRAPH_ARCH=""
BUILD_TARGET=""
DEPLOYMENT_NAME=""

if [ $# -gt 0 ]; then
    EXAMPLE_TYPE="$1"
    if [ "$EXAMPLE_TYPE" = "llm" ] && [ $# -gt 1 ]; then
        LLM_GRAPH_ARCH="$2"
    fi
else
    # Interactive selection
    info "Please select an example type:"
    select EXAMPLE_TYPE_CHOICE in "hello-world" "llm"; do
        case $EXAMPLE_TYPE_CHOICE in
            hello-world ) EXAMPLE_TYPE="hello_world"; break;;
            llm ) EXAMPLE_TYPE="llm"; break;;
            * ) error "Invalid selection. Please choose 1 or 2.";;
        esac
    done
fi

info "Selected example type: ${EXAMPLE_TYPE}"

# Handle LLM graph architecture selection
if [ "$EXAMPLE_TYPE" = "llm" ]; then
    if [ -z "$LLM_GRAPH_ARCH" ]; then
        info "Please select an LLM graph architecture:"
        # Include all available configs from the dynamo repository
        LLM_ARCH_CHOICES=("agg" "agg_router" "disagg" "disagg_router" "multinode-405b" "multinode_agg_r1" "mutinode_disagg_r1")
        select LLM_ARCH_CHOICE in "${LLM_ARCH_CHOICES[@]}"; do
            if [[ " ${LLM_ARCH_CHOICES[@]} " =~ " ${LLM_ARCH_CHOICE} " ]]; then
                LLM_GRAPH_ARCH=$LLM_ARCH_CHOICE
                break
            else
                error "Invalid selection. Please choose from the list."
            fi
        done
    fi

    # Validate LLM graph architecture
    LLM_ARCH_CHOICES=("agg" "agg_router" "disagg" "disagg_router" "multinode-405b" "multinode_agg_r1" "mutinode_disagg_r1")
    if [[ ! " ${LLM_ARCH_CHOICES[@]} " =~ " ${LLM_GRAPH_ARCH} " ]]; then
        error "Invalid LLM graph architecture: ${LLM_GRAPH_ARCH}"
        info "Available architectures: ${LLM_ARCH_CHOICES[*]}"
        exit 1
    fi

    # Set build target and config file based on architecture
    if [[ "$LLM_GRAPH_ARCH" == multinode* ]]; then
        # Multinode configs don't have corresponding graph files, they use existing graphs
        if [[ "$LLM_GRAPH_ARCH" == *"agg"* ]]; then
            BUILD_TARGET="graphs.agg:Frontend"
        else
            BUILD_TARGET="graphs.disagg:Frontend"
        fi
    else
        BUILD_TARGET="graphs.${LLM_GRAPH_ARCH}:Frontend"
    fi

    CONFIG_FILE="${SCRIPT_DIR}/dynamo/examples/llm/configs/${LLM_GRAPH_ARCH}.yaml"
    DEPLOYMENT_NAME="llm-${LLM_GRAPH_ARCH}"
    EXAMPLE_PATH="llm"
    info "Selected LLM graph architecture: ${LLM_GRAPH_ARCH}"
    info "Build target: ${BUILD_TARGET}"
    info "Config file: ${CONFIG_FILE}"
else
    BUILD_TARGET="hello_world:Frontend"
    CONFIG_FILE="${SCRIPT_DIR}/dynamo/examples/hello_world/config.yaml"
    DEPLOYMENT_NAME="hello-world"
    EXAMPLE_PATH="hello_world"
fi

# Sanitize deployment name: replace underscores with dashes
DEPLOYMENT_NAME=$(echo "${DEPLOYMENT_NAME}" | tr '_' '-')
info "Deployment name: ${DEPLOYMENT_NAME}"

# Set example directory
EXAMPLE_DIR="${SCRIPT_DIR}/dynamo/examples/${EXAMPLE_PATH}"

#---------------------------------------------------------------
# Phase 1: Build Inference Graph (5a equivalent)
#---------------------------------------------------------------

section "Phase 1: Building Inference Graph"

# Navigate to example directory
cd "${EXAMPLE_DIR}"
info "Working in example directory: ${EXAMPLE_DIR}"

# Set Docker platform for compatibility
export DOCKER_DEFAULT_PLATFORM=linux/amd64

# Check if DYNAMO_IMAGE is set, if not, provide options
if [ -z "${DYNAMO_IMAGE:-}" ]; then
    warn "DYNAMO_IMAGE environment variable not set"
    info ""

    # Check if there are existing images in ECR
    info "Checking for existing images in ECR repository..."
    EXISTING_IMAGES=$(aws ecr describe-images --repository-name "${BASE_ECR_REPOSITORY}" --region "${AWS_REGION}" --query 'imageDetails[*].imageTags[]' --output text 2>/dev/null | grep -v "None" | sort -u | head -10 || echo "")

    if [ -n "${EXISTING_IMAGES}" ]; then
        info "Found existing images in ECR:"
        echo "${EXISTING_IMAGES}" | while read -r tag; do
            echo "  - ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${BASE_ECR_REPOSITORY}:${tag}"
        done
        info ""

        info "Available images:"
        echo "${EXISTING_IMAGES}" | nl -w2 -s') '
        echo -n "Enter the number of the image to use (or 0 to exit): "
        read -r IMAGE_CHOICE

        if [ "${IMAGE_CHOICE}" = "0" ]; then
            info "Exiting..."
            exit 0
        fi

        SELECTED_TAG=$(echo "${EXISTING_IMAGES}" | sed -n "${IMAGE_CHOICE}p")
        if [ -n "${SELECTED_TAG}" ]; then
            DYNAMO_IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${BASE_ECR_REPOSITORY}:${SELECTED_TAG}"
            export DYNAMO_IMAGE
            success "Using existing image: ${DYNAMO_IMAGE}"
        else
            error "Invalid selection"
            exit 1
        fi
    else
        warn "No existing images found in ECR repository: ${BASE_ECR_REPOSITORY}"
        error ""
        error "You need to build a base image first. Please run:"
        error "  cd ${SCRIPT_DIR}"
        error "  ./build-base-image.sh vllm --push"
        error ""
        error "After the build completes, the environment file will be updated automatically."
        error "Then you can re-run this deployment script."
        exit 1
    fi
fi

success "Using DYNAMO_IMAGE: ${DYNAMO_IMAGE}"

# Build the service using the target determined from selection
info "Building ${BUILD_TARGET} service for Dynamo Cloud deployment..."
BUILD_OUTPUT=$(DYNAMO_IMAGE="${DYNAMO_IMAGE}" dynamo build "${BUILD_TARGET}" 2>&1)
echo "${BUILD_OUTPUT}"

# Extract the tag from build output (5a pattern)
info "Extracting tag from build output..."
extract_dynamo_tag() {
    local output="$1"
    echo "$output" | grep "Successfully built" | awk '{ print $3 }' | sed 's/\.$//'
}
DYNAMO_TAG=$(extract_dynamo_tag "$BUILD_OUTPUT")

if [ -z "$DYNAMO_TAG" ]; then
    error "Failed to parse DYNAMO_TAG from build output. BUILD_OUTPUT was:"
    error "$BUILD_OUTPUT"
    exit 1
fi

success "Built service with tag: ${DYNAMO_TAG}"
export DYNAMO_TAG

success "Phase 1 completed: Inference graph built"

#---------------------------------------------------------------
# Phase 2: Deploy Inference Graph (6a equivalent)
#---------------------------------------------------------------

section "Phase 2: Deploying Inference Graph"

# Update kubeconfig
info "Updating kubeconfig..."
aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}

# Deploy using Dynamo Cloud platform
info "Using Dynamo Cloud platform for deployment..."
info "Deploying service with tag: ${DYNAMO_TAG}"

# Check if config file exists and deploy accordingly (6a pattern)
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    info "Using configuration file: $CONFIG_FILE"
    info "Running: dynamo deployment create \"${DYNAMO_TAG}\" --no-wait -n \"${DEPLOYMENT_NAME}\" -f \"${CONFIG_FILE}\" --endpoint \"${DYNAMO_CLOUD}\""
    dynamo deployment create "${DYNAMO_TAG}" --no-wait -n "${DEPLOYMENT_NAME}" -f "${CONFIG_FILE}" --endpoint "${DYNAMO_CLOUD}"
else
    if [ -n "$CONFIG_FILE" ]; then
        warn "Configuration file not found: $CONFIG_FILE"
        warn "Attempting to deploy without configuration file, but this may fail."
    fi
    info "Running: dynamo deployment create \"${DYNAMO_TAG}\" --no-wait -n \"${DEPLOYMENT_NAME}\" --endpoint \"${DYNAMO_CLOUD}\""
    dynamo deployment create "${DYNAMO_TAG}" --no-wait -n "${DEPLOYMENT_NAME}" --endpoint "${DYNAMO_CLOUD}"
fi

if [ $? -eq 0 ]; then
    success "Service deployed successfully via Dynamo Cloud"
else
    error "Dynamo Cloud deployment failed"
    exit 1
fi

# Wait for deployment to be ready
info "Waiting for deployment to be ready..."
sleep 10

# Check deployment status
kubectl get pods -n ${NAMESPACE} -l app=${DEPLOYMENT_NAME} 2>/dev/null || kubectl get pods -n ${NAMESPACE}

success "Phase 2 completed: Inference graph deployed"

#---------------------------------------------------------------
# Phase 3: Create Service and ServiceMonitor
#---------------------------------------------------------------

section "Phase 3: Creating Service and ServiceMonitor"

# Function to extract port from config file
get_port_from_config() {
    local config_file="$1"
    if [ -f "$config_file" ]; then
        # Extract port from YAML config (look for Frontend.port or similar)
        local port=$(grep -E "^\s*port:\s*[0-9]+" "$config_file" | head -1 | sed 's/.*port:\s*//' | tr -d ' ')
        if [ -n "$port" ]; then
            echo "$port"
        else
            echo "8000"  # Default port
        fi
    else
        echo "3000"  # Default port if no config file
    fi
}

# Get the port from the config file used in deployment
if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
    APP_PORT=$(get_port_from_config "$CONFIG_FILE")
    info "Detected application port from config: $APP_PORT"
else
    APP_PORT="8000"
    info "Using default application port: $APP_PORT"
fi

# Create config directory for monitoring resources
CONFIG_DIR="${SCRIPT_DIR}/monitoring-config"
mkdir -p "${CONFIG_DIR}"
info "Created monitoring config directory: ${CONFIG_DIR}"

# Create Service + ServiceMonitor for frontend
info "Creating Service and ServiceMonitor configuration..."
cat > "${CONFIG_DIR}/dynamo-frontend-service-monitor.yaml" << EOF
# ServiceMonitor for frontend metrics
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ${DEPLOYMENT_NAME}-frontend-service-metrics
  namespace: monitoring
  labels:
    release: prometheus-stack
    dynamo.ai/monitoring-type: frontend-service
    dynamo.ai/deployment-name: ${DEPLOYMENT_NAME}
spec:
  selector:
    matchLabels:
      nvidia.com/selector: ${DEPLOYMENT_NAME}-frontend
  namespaceSelector:
    matchNames:
      - ${NAMESPACE}
  endpoints:
    - port: app
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
EOF

# Apply Service and ServiceMonitor (only if monitoring namespace exists)
if kubectl get namespace monitoring &>/dev/null; then
    info "Applying Frontend Service + ServiceMonitor..."
    kubectl apply -f "${CONFIG_DIR}/dynamo-frontend-service-monitor.yaml"
    success "Frontend Service + ServiceMonitor created successfully."

    info "Monitoring configured with service-based approach:"
    info "- Service: ${DEPLOYMENT_NAME}-frontend on port ${APP_PORT}"
    info "- ServiceMonitor: ${DEPLOYMENT_NAME}-frontend-service-metrics"
    info "Frontend app service provides clean access to both LLM calls and metrics endpoint."
else
    warn "Monitoring namespace not found. Monitor files created but not applied."
    info "To apply manually when monitoring is available:"
    info "kubectl apply -f ${CONFIG_DIR}/dynamo-frontend-service-monitor.yaml"
fi

success "Phase 3 completed: Service and ServiceMonitor created"

#---------------------------------------------------------------
# Phase 4: Service Exposure and Testing
#---------------------------------------------------------------

section "Phase 4: Service Exposure"

# Use the service we just created
SERVICE_NAME="${DEPLOYMENT_NAME}-frontend-app"

info "Using created service: ${SERVICE_NAME}"

# Get service details
if kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} >/dev/null 2>&1; then
    kubectl describe service ${SERVICE_NAME} -n ${NAMESPACE}

    # Set up port forwarding for testing
    LOCAL_PORT=${APP_PORT}

    info "Setting up port forwarding..."
    info "Service will be available at: http://localhost:${LOCAL_PORT}"
    info "To access the service, run:"
    echo "  kubectl port-forward service/${SERVICE_NAME} ${LOCAL_PORT}:${APP_PORT} -n ${NAMESPACE}"
else
    warn "Created service not found, checking for other services..."
    # Try to find any service in the namespace
    FALLBACK_SERVICE=$(kubectl get services -n ${NAMESPACE} -o name | head -1 | cut -d'/' -f2)
    if [ -n "${FALLBACK_SERVICE}" ]; then
        info "Found fallback service: ${FALLBACK_SERVICE}"
        kubectl describe service ${FALLBACK_SERVICE} -n ${NAMESPACE}
    else
        warn "No services found in namespace ${NAMESPACE}"
        info "Available services in namespace ${NAMESPACE}:"
        kubectl get services -n ${NAMESPACE}
    fi
fi

success "Deployment completed successfully!"

echo ""
echo "Summary:"
echo "  Example Type: ${EXAMPLE_TYPE}"
if [ "$EXAMPLE_TYPE" = "llm" ]; then
    echo "  LLM Architecture: ${LLM_GRAPH_ARCH}"
fi
echo "  Build Target: ${BUILD_TARGET}"
echo "  Dynamo Tag: ${DYNAMO_TAG}"
echo "  Deployment Name: ${DEPLOYMENT_NAME}"
echo "  Namespace: ${NAMESPACE}"
echo "  Service: ${SERVICE_NAME}"
echo "  Application Port: ${APP_PORT}"
echo "  Config Directory: ${CONFIG_DIR}"
echo ""
echo "Created Resources:"
echo "  - Service: ${DEPLOYMENT_NAME}-frontend-app"
echo "  - ServiceMonitor: ${DEPLOYMENT_NAME}-frontend-service-metrics"
echo "  - Config File: ${CONFIG_DIR}/dynamo-frontend-service-monitor.yaml"
echo ""
echo "Next steps:"
echo "1. Set up port forwarding to access the service:"
echo "   kubectl port-forward service/${DEPLOYMENT_NAME}-frontend-app ${APP_PORT}:${APP_PORT} -n ${NAMESPACE}"
echo "2. Run the test script: ./test.sh ${DEPLOYMENT_NAME}"
echo "3. Monitor the deployment: kubectl get pods -n ${NAMESPACE} -w"
echo "4. Check monitoring: kubectl get servicemonitors -n monitoring"
