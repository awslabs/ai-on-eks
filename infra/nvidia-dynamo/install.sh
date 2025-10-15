#!/bin/bash

# Prompt for NGC API key if not already set
if [ -z "$NGC_API_KEY" ]; then
  echo "NVIDIA Dynamo requires NGC (NVIDIA GPU Cloud) authentication for container images."
  echo "You can get your NGC API key from: https://ngc.nvidia.com/setup/api-key"
  echo ""
  read -p "Please enter your NGC API key: " NGC_API_KEY

  if [ -z "$NGC_API_KEY" ]; then
    echo "Error: NGC API key is required for NVIDIA Dynamo deployment"
    exit 1
  fi

  # Export for use in terraform and kubectl commands
  export NGC_API_KEY
fi

echo "Using NGC API key: ${NGC_API_KEY:0:10}..."

# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL


cd terraform/_LOCAL
source ./install.sh

# Wait for base infrastructure to be ready
echo "Waiting for infrastructure to be ready..."
sleep 30

# Update kubeconfig for kubectl access
eval "$(terraform output -raw configure_kubectl)"

# Setup NGC authentication for ArgoCD and Dynamo Platform
echo "Setting up NGC authentication..."

# Add NGC Helm repository with authentication for ArgoCD
echo "Adding NGC Helm repository to ArgoCD..."

# Create ArgoCD repository secret for NGC
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: nvidia-dynamo-repo
  namespace: argocd
  labels:
    argocd.argoproj.io/secret-type: repository
type: Opaque
stringData:
  type: helm
  name: nvidia-dynamo
  url: https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts
  username: \$oauthtoken
  password: $NGC_API_KEY
EOF

# Create NGC image pull secret for dynamo-cloud namespace
echo "Creating NGC image pull secret for dynamo-cloud namespace..."
kubectl create namespace dynamo-cloud --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret docker-registry docker-imagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  --namespace=dynamo-cloud \
  --dry-run=client -o yaml | kubectl apply -f -

echo "NVIDIA Dynamo deployment completed!"
echo ""
echo "Next steps:"
echo "1. Check ArgoCD for Dynamo platform deployment: kubectl get applications -n argocd"
echo "2. Monitor Dynamo pods: kubectl get pods -n dynamo-cloud"
echo "3. View available NodePools: kubectl get nodepools"
echo "4. Use blueprints for inference examples: cd ../../blueprints/inference/nvidia-dynamo"
echo ""
echo "NGC Authentication configured:"
echo "  - ArgoCD repository secret: nvidia-dynamo-repo (for Helm chart access)"
echo "  - Image pull secret: docker-imagepullsecret (for container image access)"
echo "  - Both secrets use NGC API key: ${NGC_API_KEY:0:10}..."
