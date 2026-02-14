#!/bin/bash

# Script to deploy model hosting services

set -e

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to display messages with timestamp
log() {
  echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
  echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
  cleanup_on_exit
  exit 1
}

success() {
  echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Cleanup function to restore original files
cleanup_on_exit() {
  log "Performing cleanup..."
  if [ -f "ray-services/ray-service-llamacpp-with-embedding.yaml.backup" ]; then
    log "Restoring original YAML file..."
    mv "ray-services/ray-service-llamacpp-with-embedding.yaml.backup" "ray-services/ray-service-llamacpp-with-embedding.yaml"
    success "Original YAML file restored!"
  fi
}

# Set up trap to cleanup on script exit
trap cleanup_on_exit EXIT

# Function to prompt for Hugging Face token
get_hugging_face_token() {
  log "Hugging Face token is required for model access..."
  
  # Check if token is already set as environment variable
  if [ -n "$HUGGING_FACE_TOKEN" ]; then
    log "Using Hugging Face token from environment variable HUGGING_FACE_TOKEN"
    HF_TOKEN="$HUGGING_FACE_TOKEN"
  else
    # Provide instructions for getting a token
    echo
    echo "To get a Hugging Face token:"
    echo "1. Go to https://huggingface.co/settings/tokens"
    echo "2. Create a new token with 'Read' permissions"
    echo "3. Copy the token and paste it below"
    echo
    echo "Alternatively, you can set the HUGGING_FACE_TOKEN environment variable:"
    echo "export HUGGING_FACE_TOKEN=your_token_here"
    echo
    
    # Prompt user for token
    echo -n "Please enter your Hugging Face token: "
    read HF_TOKEN
    
    if [ -z "$HF_TOKEN" ]; then
      error "Hugging Face token is required. Please provide a valid token."
    fi
    
    # Basic validation - HF tokens typically start with 'hf_'
    if [[ ! "$HF_TOKEN" =~ ^hf_.* ]]; then
      warn "Warning: Hugging Face tokens typically start with 'hf_'. Please verify your token is correct."
      echo -n "Continue anyway? (y/N): "
      read -r confirm
      if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        error "Token validation failed. Please provide a valid Hugging Face token."
      fi
    fi
  fi
  
  success "Hugging Face token configured successfully!"
}

# Function to update YAML files with Hugging Face token
update_yaml_with_token() {
  log "Updating YAML files with Hugging Face token..."
  
  # Update ray-service YAML file
  if [ -f "ray-services/ray-service-llamacpp-with-embedding.yaml" ]; then
    # Create a backup of the original file
    cp "ray-services/ray-service-llamacpp-with-embedding.yaml" "ray-services/ray-service-llamacpp-with-embedding.yaml.backup"
    
    # Replace the token placeholder with actual token
    sed -i.tmp "s/HUGGING_FACE_TOKEN_HERE/$HF_TOKEN/g" "ray-services/ray-service-llamacpp-with-embedding.yaml"
    rm "ray-services/ray-service-llamacpp-with-embedding.yaml.tmp" 2>/dev/null || true
    
    success "Ray service YAML updated with Hugging Face token!"
  else
    error "ray-service-llamacpp-with-embedding.yaml not found in ray-services directory"
  fi
}

# Check prerequisites
check_prerequisites() {
  log "Checking prerequisites..."
  
  # Check kubectl
  if ! command -v kubectl &> /dev/null; then
    error "kubectl is not installed. Please install it first."
  fi
  
  # Check if kubectl is configured to access a cluster
  if ! kubectl cluster-info &> /dev/null; then
    error "Cannot access Kubernetes cluster. Please check your kubeconfig."
  fi
  
  success "All prerequisites satisfied."
}

# Install Ray service with llamacpp and embedding
install_ray_llamacpp_embedding() {
  log "Installing Ray service with llamacpp and embedding..."
  
  if [ -f "ray-services/ray-service-llamacpp-with-embedding.yaml" ]; then
    kubectl apply -f ray-services/ray-service-llamacpp-with-embedding.yaml
    success "Ray service with llamacpp and embedding deployed successfully!"
  else
    error "ray-service-llamacpp-with-embedding.yaml not found in ray-services directory"
  fi
}

# Install standalone vLLM reasoning service
install_standalone_vllm_reasoning() {
  log "CHECK: Installing standalone vLLM reasoning service..."
  
  if [ -f "standalone-vllm-reasoning.yaml" ]; then
    kubectl apply -f standalone-vllm-reasoning.yaml
    success "OK! Standalone vLLM reasoning service deployed successfully!"
  else
    error "PROBLEM: standalone-vllm-reasoning.yaml not found"
  fi
}

# Install standalone vLLM vision service
install_standalone_vllm_vision() {
  log "Installing standalone vLLM vision service..."
  
  if [ -f "standalone-vllm-vision.yaml" ]; then
    kubectl apply -f standalone-vllm-vision.yaml
    success "Standalone vLLM vision service deployed successfully!"
  else
    error "standalone-vllm-vision.yaml not found"
  fi
}

# Wait for services to be ready
wait_for_services() {
  log "Waiting for services to be ready..."
  
  # Wait for Ray service
  log "Waiting for Ray service to be ready..."
  kubectl wait --for=condition=ready --timeout=600s rayservice --all 2>/dev/null || warn "Ray service might still be initializing"
  
  # Wait for standalone services
  log "Waiting for standalone services to be ready..."
  kubectl wait --for=condition=available --timeout=600s deployment --all 2>/dev/null || warn "Some deployments might still be initializing"
  
  success "Services are ready!"
}

# Verify installations
verify_installations() {
  log "Verifying installations..."
  
  log "Checking Ray services..."
  kubectl get rayservice
  
  log "Checking deployments..."
  kubectl get deployments
  
  log "Checking services..."
  kubectl get services
  
  log "Checking pods..."
  kubectl get pods
  
  success "Installation verification completed!"
}

# Main execution
main() {
  log "Starting model hosting and Agentic AI deployment on the Inference Ready EKS cluster..."
  
  check_prerequisites
  get_hugging_face_token
  update_yaml_with_token
  install_ray_llamacpp_embedding
  install_standalone_vllm_reasoning
  wait_for_services
  verify_installations
  
  success "All model hosting services deployed successfully!"
  log "You can now access your model hosting services."
}

# Execute main function
main
