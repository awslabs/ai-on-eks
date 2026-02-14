#!/bin/bash

# Script to validate existing EKS cluster and install KubeRay and NVIDIA GPU operators
#
# Storage Class Configuration:
# - Creates GP3 storage class with 'Immediate' binding mode instead of 'WaitForFirstConsumer'
# - This prevents volume binding timeout issues that can occur with StatefulSets
# - Immediate binding ensures PVs are created immediately when PVCs are created
# - Compatible with both simple deployments and complex StatefulSet configurations

set -e

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Export current EKS cluster name
export CLUSTER_NAME=$(kubectl config current-context | cut -d'/' -f2)

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

# Check prerequisites
check_prerequisites() {
  log "Checking prerequisites..."
  
  # Install gettext if not available
  if ! command -v envsubst &> /dev/null; then
    log "Installing gettext package for envsubst..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS
      if command -v brew &> /dev/null; then
        brew install gettext
        # Add gettext to PATH for current session
        export PATH="/usr/local/opt/gettext/bin:$PATH"
        # For Apple Silicon Macs
        if [[ -d "/opt/homebrew/opt/gettext/bin" ]]; then
          export PATH="/opt/homebrew/opt/gettext/bin:$PATH"
        fi
      else
        error "Homebrew is not installed. Please install Homebrew first or install gettext manually."
      fi
    elif command -v apt-get &> /dev/null; then
      # Ubuntu/Debian
      sudo apt-get update && sudo apt-get install -y gettext
    elif command -v yum &> /dev/null; then
      # RHEL/CentOS/Amazon Linux
      sudo yum install -y gettext
    elif command -v dnf &> /dev/null; then
      # Fedora
      sudo dnf install -y gettext
    else
      error "Cannot install gettext. Please install it manually for your distribution."
    fi
    success "gettext installed successfully."
  fi
  
  # Check AWS CLI
  if ! command -v aws &> /dev/null; then
    error "AWS CLI is not installed. Please install it first."
  fi
  
  # Check kubectl
  if ! command -v kubectl &> /dev/null; then
    error "kubectl is not installed. Please install it first."
  fi
  
  # Check helm
  if ! command -v helm &> /dev/null; then
    error "Helm is not installed. Please install it first."
  fi
  
  # Check AWS credentials
  if ! aws sts get-caller-identity &> /dev/null; then
    error "AWS credentials not configured or invalid. Please configure AWS CLI."
  fi
  
  success "All prerequisites satisfied."
}

# Validate existing EKS cluster
validate_eks_cluster() {
  log "Validating existing EKS cluster..."
  
  # Check if kubectl is configured to access a cluster
  if ! kubectl cluster-info &> /dev/null; then
    error "Cannot access Kubernetes cluster. Please check your kubeconfig."
  fi
  
  # Get cluster info
  CLUSTER_INFO=$(kubectl cluster-info)
  log "Connected to Kubernetes cluster:"
  echo "$CLUSTER_INFO"
  
  # Check nodes
  log "Checking cluster nodes..."
  NODE_COUNT=$(kubectl get nodes --no-headers | wc -l)
  if [ "$NODE_COUNT" -lt 1 ]; then
    error "No nodes found in the cluster. Please check your EKS cluster."
  fi
  
  log "Found $NODE_COUNT nodes in the cluster:"
  kubectl get nodes
  
  # Check for GPU nodes (optional)
  if kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com\\/gpu --no-headers | grep -v "<none>" &> /dev/null; then
    log "GPU nodes detected in the cluster."
  else
    warn "No GPU nodes detected. NVIDIA GPU operator will still be installed but may not be utilized."
  fi
  
  success "EKS cluster validation completed successfully!"
}

# Install KubeRay operator
install_kuberay_operator() {
  log "Installing KubeRay operator..."
  
  # Create namespace for KubeRay
  # kubectl create namespace kuberay --dry-run=client -o yaml | kubectl apply -f -
  
  # Add KubeRay Helm repository
  helm repo add kuberay https://ray-project.github.io/kuberay-helm/
  helm repo update
  
  # Check if KubeRay operator is already installed
  #Update for inference ready EKS cluster use kuberay-operator ns
 if helm list -n kuberay-operator | grep kuberay-operator &> /dev/null; then
    warn "KubeRay operator is already installed. Upgrading..."
    helm upgrade kuberay-operator kuberay/kuberay-operator -n kuberay-operator
  else
    # Install KubeRay operator
    log "Installing KubeRay operator from Helm chart..."
    helm install kuberay-operator kuberay/kuberay-operator -n kuberay-operator
  fi
  
  # Wait for the operator to be ready
  log "Waiting for KubeRay operator to be ready..."
  kubectl wait --for=condition=available --timeout=300s deployment/kuberay-operator -n kuberay-operator
  
  success "KubeRay operator installed successfully!"
}

# Install NVIDIA Device Plugin (standalone)
install_nvidia_device_plugin() {
  log "Installing NVIDIA Device Plugin..."
  
  # Check if NVIDIA Device Plugin is already installed
  if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &> /dev/null; then
    warn "NVIDIA Device Plugin is already installed. Skipping..."
    return
  fi
  
  # Patching cluster policy to allow creation of NVIDIA ToolKit driver folders on BottleRocket nodes:
  log "Updating Cluster policy to allow installation of NVIDIA Toolkit on BottleRocket OS nodes.."
  kubectl patch clusterpolicy cluster-policy --type merge --patch-file fix-nvidia-toolkit-bottlerocket.yaml

  # Install NVIDIA Device Plugin
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.2/deployments/static/nvidia-device-plugin.yml
  
  # Wait for device plugin to be ready
 log "Waiting for NVIDIA Device Plugin to be ready..."
 kubectl rollout status daemonset/nvidia-device-plugin-daemonset -n kube-system --timeout=300s
  
  # Verify installation
  if kubectl get pods -n kube-system | grep nvidia-device-plugin | grep Running > /dev/null; then
    success "NVIDIA Device Plugin is running successfully!"
  else
    warn "NVIDIA Device Plugin may still be starting up..."
  fi
  
  success "NVIDIA Device Plugin installation completed!"
 }

# Install NVIDIA GPU operator
install_nvidia_gpu_operator() {
  log "Installing NVIDIA GPU operator..."
  
  # Create namespace for GPU operator
  kubectl create namespace gpu-operator --dry-run=client -o yaml | kubectl apply -f -
  
  # Add NVIDIA Helm repository
  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
  helm repo update
  
  # Check if NVIDIA GPU operator is already installed
  if helm list -n gpu-operator | grep gpu-operator &> /dev/null; then
    warn "NVIDIA GPU operator is already installed. Upgrading..."
    helm upgrade -n gpu-operator $(helm list -n gpu-operator | grep gpu-operator | awk '{print $1}') nvidia/gpu-operator
  else
    # Install NVIDIA GPU operator
    helm install --wait --generate-name \
      -n gpu-operator \
      nvidia/gpu-operator
  fi
  
  # Wait for the operator to be ready
  log "Waiting for NVIDIA GPU operator to be ready..."
  kubectl wait --for=condition=available --timeout=600s -n gpu-operator deployment/gpu-operator-node-feature-discovery-master 2>/dev/null || true
  kubectl wait --for=condition=available --timeout=600s -n gpu-operator deployment/gpu-operator 2>/dev/null || true
  
  success "NVIDIA GPU operator installed successfully!"
}

# Install GP3 storage class with Immediate binding mode
# Note: Using Immediate binding mode instead of WaitForFirstConsumer to avoid
# volume binding timeout issues that can occur with StatefulSets and complex scheduling
install_gp3_storage() {
  log "Installing AutoMode compatible GP3 storage class with Immediate binding mode..."
  
  if kubectl get storageclass gp3 &> /dev/null; then
    warn "GP3 storage class already exists. Checking configuration..."
    
    # Check if the existing storage class has the correct binding mode
    BINDING_MODE=$(kubectl get storageclass gp3 -o jsonpath='{.volumeBindingMode}')
    if [ "$BINDING_MODE" != "Immediate" ]; then
      warn "Existing GP3 storage class uses '$BINDING_MODE' binding mode."
      warn "For optimal compatibility, consider updating to 'Immediate' binding mode."
      warn "You can delete the existing storage class and rerun this script to update it."
    else
      success "GP3 storage class already configured with Immediate binding mode."
    fi
  else
    kubectl apply -f gp3.yaml
    
    # Validate the storage class was created correctly
    if kubectl get storageclass gp3 &> /dev/null; then
      BINDING_MODE=$(kubectl get storageclass gp3 -o jsonpath='{.volumeBindingMode}')
      PROVISIONER=$(kubectl get storageclass gp3 -o jsonpath='{.provisioner}')
      success "GP3 storage class installed successfully!"
      log "  - Binding Mode: $BINDING_MODE"
      log "  - Provisioner: $PROVISIONER"
      log "  - Default: $(kubectl get storageclass gp3 -o jsonpath='{.metadata.annotations.storageclass\.kubernetes\.io/is-default-class}')"
    else
      error "Failed to create GP3 storage class"
    fi
  fi
}

# Install Karpenter node pools
install_karpenter_nodepools() {
  log "Installing Karpenter AutoMode node pools..."
  
  # Install all node pool configurations with environment variable substitution
  for nodepool_file in karpenter_nodepool/*.yaml; do
    if [ -f "$nodepool_file" ]; then
      log "Installing node pool: $(basename "$nodepool_file") with cluster name: $CLUSTER_NAME"
      envsubst < "$nodepool_file" | kubectl apply -f -
    fi
  done
  
  success "All Karpenter node classes/pools installed successfully!"
}

# Verify installations
verify_installations() {
  #log "Verifying KubeRay operator installation..."
  #kubectl get all -n kuberay
  
  #log "Verifying NVIDIA GPU operator installation..."
  #kubectl get all -n gpu-operator
  
  #log "Verifying NVIDIA Device Plugin installation..."
  #if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &> /dev/null; then
  #  success "NVIDIA Device Plugin daemonset is installed."
  #  kubectl get pods -n kube-system | grep nvidia-device-plugin || true
  #else
  #  warn "NVIDIA Device Plugin daemonset not found."
  #fi
  
  log "Checking for RayCluster Kubernetes CRD..."
  if kubectl get crd rayclusters.ray.io &> /dev/null; then
    success "RayCluster CRD is installed."
  else
    warn "RayCluster CRD not found. KubeRay operator might not be functioning correctly."
  fi
  
  #log "Checking for NVIDIA GPU operator components..."
  #if kubectl get pods -n gpu-operator | grep -q "nvidia-device-plugin"; then
  #  success "NVIDIA device plugin found."
  #else
  #  warn "NVIDIA device plugin not found. GPU operator might still be initializing."
  #fi
  
  log "Checking for GPU resources on nodes..."
  if kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com\\/gpu --no-headers | grep -v "<none>" &> /dev/null; then
    success "GPU resources detected on nodes:"
    kubectl get nodes -o=custom-columns=NAME:.metadata.name,GPU:.status.capacity.nvidia\\.com\\/gpu --no-headers | grep -v "<none>"
  else
    warn "No GPU resources found yet. This is normal if no GPU nodes are currently provisioned."
  fi

  
  log "Verifying Karpenter node classes and pools..."
  kubectl get nodeclasses,nodepools
}

# Main execution
main() {
  log "Starting validation and installation process..."
  log "Detected Inference EKS cluster name: $CLUSTER_NAME"
  
  check_prerequisites
  validate_eks_cluster
  #this call will not be needed when switched to Inference Ready cluster
  #install_kuberay_operator
  #this call will likely not be needed when switched to Inference Ready cluster
  # Try commenting out NVIDIA GPU operator and device plugin
  #install_nvidia_gpu_operator
  #install_nvidia_device_plugin  # Added NVIDIA device plugin installation
  install_gp3_storage
  #install_karpenter_nodepools
  verify_installations
  
  success "All Solution Kubernetes components installed successfully!"
  #log "Your AutoMode EKS cluster now has KubeRay, NVIDIA GPU operators, NVIDIA device plugin, and Karpenter Node Classes/Pools installed."
  log "✓ Your AutoMode EKS cluster has KubeRay operator, GP3 storage driver and required Karpenter auto-scaling Node Classes/Pools installed."
  log ""
  log "Storage Configuration:"
  log "  ✓ GP3 storage class configured with 'Immediate binding' mode"
  log "  ✓ This prevents volume binding timeout issues with StatefulSets"
  log "  ✓ Compatible with both simple deployments and complex workloads"
}

# Execute main function
main
