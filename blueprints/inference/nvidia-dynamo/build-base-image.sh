#!/bin/bash

#---------------------------------------------------------------
# NVIDIA Dynamo Base Image Builder
#
# This script builds and pushes Dynamo base images for different
# inference frameworks. It should be run from the blueprint
# directory after the infrastructure is set up.
#
# Usage:
#   ./build-base-image.sh [framework] [options]
#
# Frameworks:
#   vllm         - vLLM framework (default)
#   tensorrtllm  - TensorRT-LLM framework
#   sglang       - SGLang framework
#   none         - Base image without inference framework
#
# Options:
#   --tag TAG    - Image tag (default: latest)
#   --push       - Push to registry after build
#   --help       - Show this help message
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

# Show help
show_help() {
    cat << EOF
NVIDIA Dynamo Base Image Builder

Usage: $0 [framework] [options]

Frameworks:
  vllm         Build vLLM-based image (default)
  tensorrtllm  Build TensorRT-LLM-based image
  sglang       Build SGLang-based image
  none         Build base image without inference framework

Options:
  --tag TAG    Image tag (default: latest)
  --push       Push to registry after build
  --help       Show this help message

Examples:
  $0 vllm --push                    # Build and push vLLM image
  $0 tensorrtllm --tag v1.0         # Build TensorRT-LLM image with tag v1.0
  $0 none --tag base-only --push    # Build and push base-only image

Environment:
  The script uses environment variables from dynamo_env.sh:
  - AWS_ACCOUNT_ID, AWS_REGION for ECR registry
  - BASE_ECR_REPOSITORY for repository name
  - IMAGE_TAG for default tag
EOF
}

# Parse arguments
FRAMEWORK="vllm"
CUSTOM_TAG=""
PUSH_IMAGE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        vllm|tensorrtllm|sglang|none)
            FRAMEWORK="$1"
            shift
            ;;
        --tag)
            CUSTOM_TAG="$2"
            shift 2
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

print_banner "DYNAMO BASE IMAGE BUILDER"

# Load environment configuration
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
    info "Loaded environment configuration from ${ENV_FILE}"
else
    error "Environment file not found: ${ENV_FILE}"
    error "Please run the infrastructure installation first: infra/nvidia-dynamo/install.sh"
    exit 1
fi

# Set image tag
if [ -n "${CUSTOM_TAG}" ]; then
    FINAL_TAG="${CUSTOM_TAG}"
else
    FINAL_TAG="${IMAGE_TAG:-latest}"
fi

# Add framework suffix to tag
if [ "${FRAMEWORK}" != "none" ]; then
    FINAL_TAG="${FINAL_TAG}-${FRAMEWORK}"
fi

info "Framework: ${FRAMEWORK}"
info "Tag: ${FINAL_TAG}"
info "Push: ${PUSH_IMAGE}"

# Check if dynamo repository exists
if [ ! -d "${SCRIPT_DIR}/dynamo" ]; then
    error "Dynamo repository not found at ${SCRIPT_DIR}/dynamo"
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

#---------------------------------------------------------------
# Build Base Image
#---------------------------------------------------------------

section "Building Base Image"

# Navigate to container directory
cd "${SCRIPT_DIR}/dynamo/container"
info "Working in container directory: $(pwd)"

# Login to ECR
info "Logging in to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Build base image with retry logic for NIXL issues
build_base_image() {
    local max_retries=3
    local retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        info "Building base image (attempt $((retry_count + 1))/$max_retries)..."

        # Build command based on framework
        local build_cmd="./build.sh"
        if [ "${FRAMEWORK}" != "none" ]; then
            build_cmd="${build_cmd} --framework ${FRAMEWORK}"
        fi

        if ${build_cmd} 2>&1 | tee /tmp/build_output.log; then
            info "Base image build completed successfully!"
            return 0
        else
            if grep -q "Failed to checkout NIXL commit.*The cached directory may be out of date" /tmp/build_output.log; then
                warn "NIXL checkout error detected. Cleaning up and retrying..."
                rm -rf /tmp/nixl
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $max_retries ]; then
                    sleep 2
                else
                    error "Build failed after $max_retries attempts"
                    exit 1
                fi
            else
                error "Build failed with non-NIXL error"
                exit 1
            fi
        fi
    done
}

# Build the image
build_base_image
rm -f /tmp/build_output.log

# Tag the image
LOCAL_IMAGE_NAME="dynamo:latest"
if [ "${FRAMEWORK}" != "none" ]; then
    LOCAL_IMAGE_NAME="dynamo:latest-${FRAMEWORK}"
fi

REGISTRY_IMAGE_NAME="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${BASE_ECR_REPOSITORY}:${FINAL_TAG}"

info "Tagging image: ${LOCAL_IMAGE_NAME} -> ${REGISTRY_IMAGE_NAME}"
docker tag "${LOCAL_IMAGE_NAME}" "${REGISTRY_IMAGE_NAME}"

# Push if requested
if [ "${PUSH_IMAGE}" = true ]; then
    info "Pushing image to registry..."
    docker push "${REGISTRY_IMAGE_NAME}"
    success "Image pushed successfully: ${REGISTRY_IMAGE_NAME}"
else
    info "Image built locally: ${REGISTRY_IMAGE_NAME}"
    info "To push the image, run: docker push ${REGISTRY_IMAGE_NAME}"
fi

success "Base image build completed!"

# Update environment file with DYNAMO_IMAGE if image was pushed
if [ "${PUSH_IMAGE}" = true ]; then
    info "Updating environment file with DYNAMO_IMAGE..."

    # Check if DYNAMO_IMAGE is already in the environment file
    if grep -q "^export DYNAMO_IMAGE=" "${ENV_FILE}"; then
        # Update existing line
        sed -i "s|^export DYNAMO_IMAGE=.*|export DYNAMO_IMAGE=\"${REGISTRY_IMAGE_NAME}\"|" "${ENV_FILE}"
        success "Updated DYNAMO_IMAGE in ${ENV_FILE}"
    else
        # Add new line
        echo "" >> "${ENV_FILE}"
        echo "# Dynamo base image for inference graphs" >> "${ENV_FILE}"
        echo "export DYNAMO_IMAGE=\"${REGISTRY_IMAGE_NAME}\"" >> "${ENV_FILE}"
        success "Added DYNAMO_IMAGE to ${ENV_FILE}"
    fi

    # Also export for current session
    export DYNAMO_IMAGE="${REGISTRY_IMAGE_NAME}"
    success "DYNAMO_IMAGE exported for current session"
fi

echo ""
echo "Summary:"
echo "  Framework: ${FRAMEWORK}"
echo "  Local image: ${LOCAL_IMAGE_NAME}"
echo "  Registry image: ${REGISTRY_IMAGE_NAME}"
echo "  Pushed: ${PUSH_IMAGE}"
if [ "${PUSH_IMAGE}" = true ]; then
    echo "  Environment file updated: Yes"
fi
echo ""
if [ "${PUSH_IMAGE}" = true ]; then
    echo "The DYNAMO_IMAGE environment variable has been set automatically."
    echo "You can now run inference graph deployments:"
    echo "  ./deploy.sh"
else
    echo "To use this image in your inference graphs:"
    echo "  export DYNAMO_IMAGE=${REGISTRY_IMAGE_NAME}"
    echo "  dynamo build your_graph:Service --containerize"
fi
