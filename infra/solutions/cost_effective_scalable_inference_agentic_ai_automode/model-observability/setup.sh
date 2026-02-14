#!/bin/bash

# Script to deploy model observability services

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

# Function to check if Langfuse is already deployed
check_existing_deployment() {
  if kubectl get pods -l app.kubernetes.io/instance=langfuse --no-headers 2>/dev/null | grep -q "Running"; then
    warn "Langfuse appears to be already running. Checking deployment status..."
    kubectl get pods -l app.kubernetes.io/instance=langfuse
    return 0
  fi
  return 1
}

log "Checking for existing Langfuse deployment..."
if check_existing_deployment; then
  log "Langfuse is already deployed and running. Skipping deployment steps..."
else
  log "Installing Langfuse secrets..."
  if kubectl apply -f langfuse-secret.yaml --dry-run=client > /dev/null 2>&1; then
    kubectl apply -f langfuse-secret.yaml
    success "Langfuse secrets applied successfully!"
  else
    error "Failed to validate langfuse-secret.yaml"
  fi

  log "Adding Langfuse Helm repository..."
  helm repo add langfuse https://langfuse.github.io/langfuse-k8s
  helm repo update
  success "Langfuse Helm repository added and updated!"

  log "Installing Langfuse using Helm..."
  if [ -f "langfuse-value.yaml" ]; then
    helm install langfuse langfuse/langfuse -f langfuse-value.yaml \
      --set nodeSelector."kubernetes\.io/arch"=arm64
    success "Langfuse Helm installation initiated with arm64 node selector!"
  else
    error "langfuse-value.yaml not found"
  fi

  log "Creating Redis port configuration patch..."
  cat <<EOF > langfuse-redis-port-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langfuse-web
  namespace: default
spec:
  template:
    spec:
      containers:
      - name: langfuse-web
        env:
        - name: REDIS_PORT
          value: "6379"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langfuse-worker
  namespace: default
spec:
  template:
    spec:
      containers:
      - name: langfuse-worker
        env:
        - name: REDIS_PORT
          value: "6379"
EOF
  success "Redis port configuration patch created!"

  log "Applying Redis port configuration patch..."
  kubectl apply -f langfuse-redis-port-patch.yaml
  success "Redis port configuration patch applied!"

  log "Waiting for Langfuse pods to be running (timeout: 15 minutes)..."
  # Wait for pods to be in Running state instead of Ready condition
  start_time=$(date +%s)
  timeout=900  # 15 minutes in seconds
  
  while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -gt $timeout ]; then
      warn "Timeout reached after 15 minutes, but continuing with deployment"
      break
    fi
    
    # Count running pods vs total pods
    total_pods=$(kubectl get pods --selector=app.kubernetes.io/instance=langfuse --no-headers | wc -l)
    running_pods=$(kubectl get pods --selector=app.kubernetes.io/instance=langfuse --no-headers | grep -c "Running" || echo "0")
    
    if [ "$total_pods" -gt 0 ] && [ "$running_pods" -gt 0 ]; then
      log "$running_pods out of $total_pods Langfuse pods are running"
      success "Langfuse has running pods - continuing with deployment"
      break
    fi
    
    log "Waiting for Langfuse pods to start running ($elapsed seconds elapsed)..."
    sleep 10
  done
fi

log "Installing Langfuse web ingress..."
if [ -f "langfuse-web-ingress.yaml" ]; then
  if kubectl apply -f langfuse-web-ingress.yaml; then
    success "Langfuse web ingress installed successfully!"
  else
    warn "Failed to install Langfuse web ingress, but continuing..."
  fi
else
  warn "langfuse-web-ingress.yaml not found, skipping ingress installation"
fi

log "Verifying Langfuse installation..."
kubectl get pods -l app.kubernetes.io/instance=langfuse
kubectl get service -l app.kubernetes.io/instance=langfuse
kubectl get ingress langfuse-web-ingress-alb 2>/dev/null || warn "Langfuse ingress not found"

success "Model observability setup completed!"
log "Refer to README.md to access Langfuse and define Public/Private Keys"
