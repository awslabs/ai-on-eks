#!/bin/bash

# Script to deploy model gateway services

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
  exit 1
}

success() {
  echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Collect Langfuse configuration
collect_langfuse_config() {
  log "Configuring Langfuse integration..."
  
  echo ""
  echo "=============================================================================="
  echo "                        LANGFUSE CONFIGURATION REQUIRED"
  echo "=============================================================================="
  echo ""
  
  # Auto-detect Langfuse deployment in cluster
  log "Checking for Langfuse deployment in your cluster..."
  
  LANGFUSE_SERVICE=$(kubectl get svc -l app.kubernetes.io/name=langfuse --no-headers 2>/dev/null | grep langfuse-web | head -1)
  DETECTED_LANGFUSE_HOST=""
  
  if [ -n "$LANGFUSE_SERVICE" ]; then
    success "Found Langfuse web service in your cluster!"
    kubectl get svc langfuse-web
    echo ""
    
    # Check for ingress first (more likely to be externally accessible)
    LANGFUSE_INGRESS_HOST=$(kubectl get ingress langfuse-web-ingress-alb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    if [ -n "$LANGFUSE_INGRESS_HOST" ]; then
      DETECTED_LANGFUSE_HOST="http://$LANGFUSE_INGRESS_HOST"
      success "Detected Langfuse Host URL (via ingress): $DETECTED_LANGFUSE_HOST"
    fi
    
    # Check for LoadBalancer service as alternative (handle errors gracefully)
    LANGFUSE_LB=$(kubectl get svc -l app.kubernetes.io/name=langfuse,type=LoadBalancer -o jsonpath='{.items[0].status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    if [ -n "$LANGFUSE_LB" ]; then
      if [ -z "$DETECTED_LANGFUSE_HOST" ]; then
        DETECTED_LANGFUSE_HOST="http://$LANGFUSE_LB"
      fi
      log "Alternative Langfuse URL (via LoadBalancer): http://$LANGFUSE_LB"
    fi
    
    if [ -z "$DETECTED_LANGFUSE_HOST" ]; then
      warn "Langfuse service found but no external access configured."
      log "You may need to set up port-forwarding: kubectl port-forward svc/langfuse-web 3000:3000"
    fi
  else
    warn "No Langfuse service found in your cluster."
    log "Please deploy Langfuse first using: make setup-observability"
  fi
  
  echo ""
  echo "SETUP INSTRUCTIONS:"
  echo "1. 🌐 ACCESS LANGFUSE:"
  if [ -n "$DETECTED_LANGFUSE_HOST" ]; then
    echo "   Open your browser and go to: $DETECTED_LANGFUSE_HOST"
  else
    echo "   Use your Langfuse URL (e.g., https://cloud.langfuse.com)"
    echo "   Or set up port-forwarding: kubectl port-forward svc/langfuse-web 3000:3000"
    echo "   Then access: http://localhost:3000"
  fi
  echo ""
  echo "2. 🏢 CREATE ORGANIZATION: Name it 'test'"
  echo "3. 📁 CREATE PROJECT: Name it 'demo' inside the organization"
  echo "4. 🔑 GET API KEYS: Go to Settings → API Keys → Create new API key"
  echo "   - Copy the Public Key (starts with 'pk-')"
  echo "   - Copy the Secret Key (starts with 'sk-')"
  echo ""
  echo "=============================================================================="
  echo ""
  
  # Prompt for Langfuse Host with auto-detected default
  while [ -z "$LANGFUSE_HOST" ]; do
    if [ -n "$DETECTED_LANGFUSE_HOST" ]; then
      echo "Enter your Langfuse Host URL (press Enter to use detected URL):"
      echo "Detected: $DETECTED_LANGFUSE_HOST"
      echo ""
      read -p "Langfuse Host URL [$DETECTED_LANGFUSE_HOST]: " LANGFUSE_HOST
      
      # Use detected URL if nothing entered
      if [ -z "$LANGFUSE_HOST" ]; then
        LANGFUSE_HOST="$DETECTED_LANGFUSE_HOST"
      fi
    else
      echo "Enter your Langfuse Host URL:"
      echo "Examples:"
      echo "  - For Langfuse Cloud: https://cloud.langfuse.com"
      echo "  - For self-hosted: http://your-langfuse-loadbalancer.region.elb.amazonaws.com"
      echo "  - For port-forwarding: http://localhost:3000"
      echo ""
      read -p "Langfuse Host URL: " LANGFUSE_HOST
    fi
    
    if [ -z "$LANGFUSE_HOST" ]; then
      warn "Langfuse Host URL cannot be empty. Please try again."
    elif [[ ! "$LANGFUSE_HOST" =~ ^https?:// ]]; then
      warn "Please include http:// or https:// in the URL."
      LANGFUSE_HOST=""
    fi
  done
  
  # Prompt for Langfuse Public Key with validation
  while [ -z "$LANGFUSE_PUBLIC_KEY" ]; do
    echo ""
    echo "Enter your Langfuse Public Key:"
    echo "  - Should start with 'pk-'"
    echo "  - Found in Langfuse Settings → API Keys"
    echo ""
    read -p "Langfuse Public Key: " LANGFUSE_PUBLIC_KEY
    
    if [ -z "$LANGFUSE_PUBLIC_KEY" ]; then
      warn "Langfuse Public Key cannot be empty. Please try again."
    elif [[ ! "$LANGFUSE_PUBLIC_KEY" =~ ^pk- ]]; then
      warn "Public Key should start with 'pk-'. Please check and try again."
      LANGFUSE_PUBLIC_KEY=""
    fi
  done
  
  # Prompt for Langfuse Secret Key with validation
  while [ -z "$LANGFUSE_SECRET_KEY" ]; do
    echo ""
    echo "Enter your Langfuse Secret Key:"
    echo "  - Should start with 'sk-'"
    echo "  - Found in Langfuse Settings → API Keys"
    echo "  - Input will be hidden for security"
    echo ""
    read -s -p "Langfuse Secret Key: " LANGFUSE_SECRET_KEY
    echo ""
    
    if [ -z "$LANGFUSE_SECRET_KEY" ]; then
      warn "Langfuse Secret Key cannot be empty. Please try again."
    elif [[ ! "$LANGFUSE_SECRET_KEY" =~ ^sk- ]]; then
      warn "Secret Key should start with 'sk-'. Please check and try again."
      LANGFUSE_SECRET_KEY=""
    fi
  done
  
  echo ""
  echo "=============================================================================="
  echo "CONFIGURATION SUMMARY:"
  echo "  Host: $LANGFUSE_HOST"
  echo "  Public Key: ${LANGFUSE_PUBLIC_KEY:0:10}..."
  echo "  Secret Key: ${LANGFUSE_SECRET_KEY:0:10}..."
  echo "=============================================================================="
  echo ""
  
  # Auto-proceed without confirmation
  log "Proceeding with configuration..."
  
  success "Langfuse configuration collected successfully!"
}

# Update deployment YAML with Langfuse configuration
update_deployment_config() {
  log "Updating deployment configuration with Langfuse settings..."
  
  # Create a backup of the original file
  cp litellm-deployment.yaml litellm-deployment.yaml.backup
  
  # Use awk for more reliable replacement (works on both macOS and Linux)
  awk -v secret_key="$LANGFUSE_SECRET_KEY" -v public_key="$LANGFUSE_PUBLIC_KEY" -v host="$LANGFUSE_HOST" '
  /- name: LANGFUSE_SECRET_KEY/ { 
    print $0; 
    getline; 
    print "          value: \"" secret_key "\""; 
    next 
  }
  /- name: LANGFUSE_PUBLIC_KEY/ { 
    print $0; 
    getline; 
    print "          value: \"" public_key "\""; 
    next 
  }
  /- name: LANGFUSE_HOST/ { 
    print $0; 
    getline; 
    print "          value: \"" host "\""; 
    next 
  }
  { print }
  ' litellm-deployment.yaml.backup > litellm-deployment.yaml.tmp && mv litellm-deployment.yaml.tmp litellm-deployment.yaml
  
  success "Deployment configuration updated with Langfuse settings!"
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

# Install LiteLLM deployment
install_litellm_deployment() {
  log "Installing LiteLLM deployment..."
  
  if [ -f "litellm-deployment.yaml" ]; then
    kubectl apply -f litellm-deployment.yaml
    success "LiteLLM deployment applied successfully!"
  else
    error "litellm-deployment.yaml not found"
  fi
}

# Wait for LiteLLM service to be running
wait_for_litellm_service() {
  log "Waiting for LiteLLM service to be ready..."
  
  # Wait for deployment to be available
  kubectl wait --for=condition=available --timeout=600s deployment/litellm 2>/dev/null || {
    warn "LiteLLM deployment might still be initializing. Checking pod status..."
    kubectl get pods -l app=litellm
  }
  
  # Check if pods are running
  local max_attempts=30
  local attempt=1
  
  while [ $attempt -le $max_attempts ]; do
    local running_pods=$(kubectl get pods -l app=litellm --field-selector=status.phase=Running --no-headers | wc -l)
    
    if [ "$running_pods" -gt 0 ]; then
      success "LiteLLM service is running!"
      kubectl get pods -l app=litellm
      return 0
    fi
    
    log "Attempt $attempt/$max_attempts: Waiting for LiteLLM pods to be running..."
    sleep 10
    ((attempt++))
  done
  
  error "LiteLLM service failed to start within the expected time. Please check the logs."
}

# Install LiteLLM ingress
install_litellm_ingress() {
  log "Installing LiteLLM ingress..."
  
  if [ -f "litellm-ingress.yaml" ]; then
    kubectl apply -f litellm-ingress.yaml
    success "LiteLLM ingress applied successfully!"
  else
    error "litellm-ingress.yaml not found"
  fi
}

# Verify installations
verify_installations() {
  log "Verifying installations..."
  
  log "Checking LiteLLM deployment..."
  kubectl get deployment litellm
  
  log "Checking LiteLLM service..."
  kubectl get service litellm
  
  log "Checking LiteLLM ingress..."
  kubectl get ingress litellm-ingress 2>/dev/null || log "No ingress found (this is normal if ingress is not configured)"
  
  log "Checking LiteLLM pods..."
  kubectl get pods -l app=litellm
  
  # Get ingress URL if available
  local ingress_url=$(kubectl get ingress litellm-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
  if [ -n "$ingress_url" ]; then
    log "LiteLLM is accessible at: https://$ingress_url"
  else
    log "Ingress URL not yet available. It may take a few minutes for the load balancer to be provisioned."
  fi
  
  success "Installation verification completed!"
}

# Main execution
main() {
  log "Starting model gateway deployment..."
  
  check_prerequisites
  collect_langfuse_config
  update_deployment_config
  install_litellm_deployment
  wait_for_litellm_service
  install_litellm_ingress
  verify_installations
  
  success "Model gateway deployed successfully!"
  log "LiteLLM proxy is now available and can route requests to your model backends."
  log "Langfuse integration has been configured for observability and monitoring."
}

# Execute main function
main
