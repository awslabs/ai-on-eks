#!/bin/bash

# Build script for separate Docker images with ECR integration and automatic Kubernetes manifest updates
set -e

# Detect OS and set sed command for cross-platform compatibility
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS (BSD sed)
    SED_INPLACE() { sed -i "" "$@"; }
    OS_NAME="macOS"
else
    # Linux (GNU sed)
    SED_INPLACE() { sed -i "$@"; }
    OS_NAME="Linux"
fi

echo "Detected OS: ${OS_NAME}"
echo "Using appropriate sed syntax for cross-platform compatibility"
echo ""

# Configuration
AWS_REGION="us-east-1"  # Change to your preferred region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
TAG="latest"

# Repository names
MAIN_REPO_NAME="strandsdk-agentic-rag-main"
MCP_REPO_NAME="strandsdk-agentic-rag-mcp"

echo "Building Strands SDK Agentic RAG Docker Images with ECR Integration..."
echo "======================================================================"
echo "AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "AWS Region: ${AWS_REGION}"
echo "ECR Registry: ${ECR_REGISTRY}"
echo ""

# Function to create ECR repository if it doesn't exist
create_ecr_repo() {
    local repo_name=$1
    echo "Checking if ECR repository '${repo_name}' exists..."
    
    if aws ecr describe-repositories --repository-names ${repo_name} --region ${AWS_REGION} >/dev/null 2>&1; then
        echo "✅ ECR repository '${repo_name}' already exists"
    else
        echo "Creating ECR repository '${repo_name}'..."
        aws ecr create-repository \
            --repository-name ${repo_name} \
            --region ${AWS_REGION} \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        echo "✅ ECR repository '${repo_name}' created successfully"
    fi
}

# Authenticate Docker to ECR
echo "Authenticating Docker to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
echo "✅ Docker authenticated to ECR successfully"

# Create ECR repositories
create_ecr_repo ${MAIN_REPO_NAME}
create_ecr_repo ${MCP_REPO_NAME}

echo ""

# Build Main Application Image
echo "Building Main Application Image..."
docker build -f Dockerfile.main -t ${MAIN_REPO_NAME}:${TAG} .
docker tag ${MAIN_REPO_NAME}:${TAG} ${ECR_REGISTRY}/${MAIN_REPO_NAME}:${TAG}
echo "✅ Main Application Image built and tagged successfully"

# Build MCP Server Image
echo "Building MCP Server Image..."
docker build -f Dockerfile.mcp -t ${MCP_REPO_NAME}:${TAG} .
docker tag ${MCP_REPO_NAME}:${TAG} ${ECR_REGISTRY}/${MCP_REPO_NAME}:${TAG}
echo "✅ MCP Server Image built and tagged successfully"

echo ""
echo "Images built and tagged successfully:"
echo "- ${ECR_REGISTRY}/${MAIN_REPO_NAME}:${TAG}"
echo "- ${ECR_REGISTRY}/${MCP_REPO_NAME}:${TAG}"
echo ""

# Push images to ECR
echo "Pushing Main Application Image to ECR..."
docker push ${ECR_REGISTRY}/${MAIN_REPO_NAME}:${TAG}
echo "✅ Main Application Image pushed to ECR successfully"

echo "Pushing MCP Server Image to ECR..."
docker push ${ECR_REGISTRY}/${MCP_REPO_NAME}:${TAG}
echo "✅ MCP Server Image pushed to ECR successfully"

echo ""
echo "All images pushed to ECR successfully!"
echo ""

# Update Kubernetes deployment files with ECR image names
echo "Updating Kubernetes deployment files with ECR image URLs..."

# Update main app deployment
if [ -f "k8s/main-app-deployment.yaml" ]; then
    SED_INPLACE "s|image: .*strandsdk-agentic-rag-main:.*|image: ${ECR_REGISTRY}/${MAIN_REPO_NAME}:${TAG}|g" k8s/main-app-deployment.yaml
    echo "✅ Updated main-app-deployment.yaml with ECR image URL"
else
    echo "⚠️  k8s/main-app-deployment.yaml not found"
fi

# Update MCP server deployment
if [ -f "k8s/tavily-mcp-deployment.yaml" ]; then
    SED_INPLACE "s|image: .*strandsdk-agentic-rag-mcp:.*|image: ${ECR_REGISTRY}/${MCP_REPO_NAME}:${TAG}|g" k8s/tavily-mcp-deployment.yaml
    echo "✅ Updated tavily-mcp-deployment.yaml with ECR image URL"
else
    echo "⚠️  k8s/tavily-mcp-deployment.yaml not found"
fi

echo ""
echo "✅ Kubernetes deployment files updated with ECR image URLs!"
echo ""

# Clean up local images to save space
echo "Cleaning up local Docker images..."
docker rmi ${MAIN_REPO_NAME}:${TAG} ${ECR_REGISTRY}/${MAIN_REPO_NAME}:${TAG} || true
docker rmi ${MCP_REPO_NAME}:${TAG} ${ECR_REGISTRY}/${MCP_REPO_NAME}:${TAG} || true
echo "✅ Local Docker images cleaned up"

echo ""
echo "======================================================================"
echo "✅ BUILD AND DEPLOYMENT PREPARATION COMPLETE!"
echo "======================================================================"
echo ""
echo "ECR Repositories created:"
echo "- ${ECR_REGISTRY}/${MAIN_REPO_NAME}"
echo "- ${ECR_REGISTRY}/${MCP_REPO_NAME}"
echo ""
echo "Images pushed to ECR:"
echo "- ${ECR_REGISTRY}/${MAIN_REPO_NAME}:${TAG}"
echo "- ${ECR_REGISTRY}/${MCP_REPO_NAME}:${TAG}"
echo ""
echo "Next steps:"
echo "1. Update your ConfigMap secrets with actual values:"
echo "   kubectl apply -f k8s/configmap.yaml"
echo ""
echo "2. Deploy the applications:"
echo "   kubectl apply -f k8s/tavily-mcp-deployment.yaml"
echo "   kubectl apply -f k8s/main-app-deployment.yaml"
echo ""
echo "3. Check deployment status:"
echo "   kubectl get pods -l app=tavily-mcp-server"
echo "   kubectl get pods -l app=strandsdk-rag-app"
echo ""
echo "4. Get the ALB endpoint:"
echo "   kubectl get ingress strandsdk-rag-ingress-alb"
echo ""
echo "Current ECR image URLs in deployments:"
if [ -f "k8s/main-app-deployment.yaml" ]; then
    echo "Main App: $(grep 'image:' k8s/main-app-deployment.yaml | head -1 | awk '{print $2}')"
fi
if [ -f "k8s/tavily-mcp-deployment.yaml" ]; then
    echo "MCP Server: $(grep 'image:' k8s/tavily-mcp-deployment.yaml | head -1 | awk '{print $2}')"
fi
