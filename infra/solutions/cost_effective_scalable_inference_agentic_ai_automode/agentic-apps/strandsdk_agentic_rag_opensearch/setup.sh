#!/bin/bash

# Multi-Agent RAG with Strands SDK and OpenSearch - Setup Script
# This script automates the complete deployment process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get user input with default value
get_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    
    if [ -n "$default" ]; then
        read -p "$prompt [$default]: " input
        if [ -z "$input" ]; then
            input="$default"
        fi
    else
        read -p "$prompt: " input
        while [ -z "$input" ]; do
            echo "This field is required."
            read -p "$prompt: " input
        done
    fi
    
    eval "$var_name='$input'"
}

# Function to get sensitive input (visible)
get_secret_input() {
    local prompt="$1"
    local var_name="$2"
    
    read -p "$prompt: " input
    while [ -z "$input" ]; do
        echo "This field is required."
        read -p "$prompt: " input
    done
    
    eval "$var_name='$input'"
}

# Function to encode base64
encode_base64() {
    echo -n "$1" | base64
}

# Function to check existing ingress
check_existing_ingress() {
    print_status "Checking for existing ingress resources..."
    
    if kubectl get ingress strandsdk-rag-ingress-alb >/dev/null 2>&1; then
        EXISTING_INGRESS=$(kubectl get ingress strandsdk-rag-ingress-alb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
        if [ -n "$EXISTING_INGRESS" ]; then
            print_status "Found existing ingress endpoint: $EXISTING_INGRESS"
            return 0
        fi
    fi
    return 1
}

# Function to get LiteLLM ingress endpoint
get_litellm_ingress() {
    # Check for ingresses containing "litellm" in the name
    local litellm_ingresses=$(kubectl get ingress -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -i litellm || echo "")
    
    if [ -n "$litellm_ingresses" ]; then
        for ingress_name in $litellm_ingresses; do
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                echo "http://$endpoint/v1"
                return 0
            fi
        done
    fi
    
    # Fallback: Check for common LiteLLM ingress names
    local litellm_ingress_names=("litellm-ingress" "litellm-alb" "litellm-loadbalancer" "litellm")
    
    for ingress_name in "${litellm_ingress_names[@]}"; do
        if kubectl get ingress "$ingress_name" >/dev/null 2>&1; then
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                echo "http://$endpoint/v1"
                return 0
            fi
        fi
    done
    
    # Check for services with LoadBalancer type
    local litellm_services=$(kubectl get svc -o jsonpath='{.items[?(@.spec.type=="LoadBalancer")].metadata.name}' | grep -i litellm || echo "")
    if [ -n "$litellm_services" ]; then
        for service in $litellm_services; do
            local endpoint=$(kubectl get svc "$service" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                echo "http://$endpoint/v1"
                return 0
            fi
        done
    fi
    
    return 1
}

# Function to log LiteLLM detection
log_litellm_detection() {
    print_status "Checking for LiteLLM service ingress..."
    
    # Check for ingresses containing "litellm" in the name
    local litellm_ingresses=$(kubectl get ingress -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -i litellm || echo "")
    
    if [ -n "$litellm_ingresses" ]; then
        for ingress_name in $litellm_ingresses; do
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                print_status "Found LiteLLM ingress '$ingress_name': $endpoint"
                return 0
            fi
        done
    fi
    
    # Check other sources and log appropriately
    local litellm_ingress_names=("litellm-ingress" "litellm-alb" "litellm-loadbalancer" "litellm")
    
    for ingress_name in "${litellm_ingress_names[@]}"; do
        if kubectl get ingress "$ingress_name" >/dev/null 2>&1; then
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                print_status "Found LiteLLM ingress '$ingress_name': $endpoint"
                return 0
            fi
        fi
    done
    
    local litellm_services=$(kubectl get svc -o jsonpath='{.items[?(@.spec.type=="LoadBalancer")].metadata.name}' | grep -i litellm || echo "")
    if [ -n "$litellm_services" ]; then
        for service in $litellm_services; do
            local endpoint=$(kubectl get svc "$service" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                print_status "Found LiteLLM LoadBalancer service '$service': $endpoint"
                return 0
            fi
        done
    fi
    
    return 1
}

# Function to get Langfuse ingress endpoint
get_langfuse_ingress() {
    # Check for ingresses containing "langfuse" in the name
    local langfuse_ingresses=$(kubectl get ingress -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -i langfuse || echo "")
    
    if [ -n "$langfuse_ingresses" ]; then
        for ingress_name in $langfuse_ingresses; do
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                echo "http://$endpoint"
                return 0
            fi
        done
    fi
    
    # Fallback: Check for common Langfuse ingress names
    local langfuse_ingress_names=("langfuse-ingress" "langfuse-alb" "langfuse-loadbalancer" "langfuse")
    
    for ingress_name in "${langfuse_ingress_names[@]}"; do
        if kubectl get ingress "$ingress_name" >/dev/null 2>&1; then
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                echo "http://$endpoint"
                return 0
            fi
        fi
    done
    
    # Check for services with LoadBalancer type
    local langfuse_services=$(kubectl get svc -o jsonpath='{.items[?(@.spec.type=="LoadBalancer")].metadata.name}' | grep -i langfuse || echo "")
    if [ -n "$langfuse_services" ]; then
        for service in $langfuse_services; do
            local endpoint=$(kubectl get svc "$service" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                echo "http://$endpoint"
                return 0
            fi
        done
    fi
    
    return 1
}

# Function to log Langfuse detection
log_langfuse_detection() {
    print_status "Checking for Langfuse service ingress..."
    
    # Check for ingresses containing "langfuse" in the name
    local langfuse_ingresses=$(kubectl get ingress -o jsonpath='{.items[*].metadata.name}' | tr ' ' '\n' | grep -i langfuse || echo "")
    
    if [ -n "$langfuse_ingresses" ]; then
        for ingress_name in $langfuse_ingresses; do
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                print_status "Found Langfuse ingress '$ingress_name': $endpoint"
                return 0
            fi
        done
    fi
    
    # Check other sources and log appropriately
    local langfuse_ingress_names=("langfuse-ingress" "langfuse-alb" "langfuse-loadbalancer" "langfuse")
    
    for ingress_name in "${langfuse_ingress_names[@]}"; do
        if kubectl get ingress "$ingress_name" >/dev/null 2>&1; then
            local endpoint=$(kubectl get ingress "$ingress_name" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                print_status "Found Langfuse ingress '$ingress_name': $endpoint"
                return 0
            fi
        fi
    done
    
    local langfuse_services=$(kubectl get svc -o jsonpath='{.items[?(@.spec.type=="LoadBalancer")].metadata.name}' | grep -i langfuse || echo "")
    if [ -n "$langfuse_services" ]; then
        for service in $langfuse_services; do
            local endpoint=$(kubectl get svc "$service" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
            if [ -n "$endpoint" ]; then
                print_status "Found Langfuse LoadBalancer service '$service': $endpoint"
                return 0
            fi
        done
    fi
    
    return 1
}



# Function to create knowledge-base ConfigMap
create_knowledge_base_configmap() {
    print_status "Creating empty knowledge-base ConfigMap..."
    
    # Create temporary knowledge-base configmap file with empty knowledge base
    cat > /tmp/knowledge-base-configmap.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: knowledge-base
  labels:
    app: strandsdk-rag
data:
  README.md: |
    # Knowledge Base
    This ConfigMap contains knowledge base documents for the RAG application.
    Add your knowledge documents as data entries in this ConfigMap.
    
    ## Usage
    The application will automatically process these documents and create embeddings
    for retrieval-augmented generation (RAG) functionality.
    
    ## Adding Knowledge
    To add knowledge documents, update this ConfigMap with your files:
    kubectl patch configmap knowledge-base --patch '{"data":{"your-file.md":"your content here"}}'
EOF
    
    # Apply the knowledge-base ConfigMap
    kubectl apply -f /tmp/knowledge-base-configmap.yaml
    rm /tmp/knowledge-base-configmap.yaml
    print_status "Empty knowledge-base ConfigMap created successfully"
}

# Function to update configmap
update_configmap() {
    local opensearch_endpoint="$1"
    local litellm_base_url="$2"
    local embedding_base_url="$3"
    local langfuse_host="$4"
    
    print_status "Updating ConfigMap with provided values..."
    
    # Create temporary configmap file based on the existing structure
    cat > /tmp/configmap-temp.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  labels:
    app: strandsdk-rag
data:
  .env: |
    # LiteLLM Configuration for Reasoning Models
    LITELLM_BASE_URL=$litellm_base_url
    REASONING_MODEL=vllm-server-qwen3
    
    # Embedding Configuration (separate from reasoning)
    EMBEDDING_BASE_URL=$embedding_base_url
    EMBEDDING_MODEL=llamacpp-embedding
    
    # AWS Configuration  
    AWS_REGION=us-east-1
    OPENSEARCH_ENDPOINT=$opensearch_endpoint
    
    # Tavily MCP Service Configuration
    TAVILY_MCP_SERVICE_URL=http://tavily-mcp-service:8001/mcp
    
    # Optional: Langfuse for observability
    LANGFUSE_HOST=$langfuse_host
    
    # Application Settings
    KNOWLEDGE_DIR=knowledge
    OUTPUT_DIR=output
    VECTOR_INDEX_NAME=knowledge-embeddings
    TOP_K_RESULTS=3
  
  # Individual config values for environment variables
  litellm-base-url: "$litellm_base_url"
  reasoning-model: "vllm-server-qwen3"
  embedding-base-url: "$embedding_base_url"
  embedding-model: "llamacpp-embedding"
  opensearch-endpoint: "$opensearch_endpoint"
  aws-region: "us-east-1"
  vector-index-name: "knowledge-embeddings"
  tavily-mcp-service-url: "http://tavily-mcp-service:8001/mcp"
  langfuse-host: "$langfuse_host"
  knowledge-dir: "knowledge"
  output-dir: "output"
  top-k-results: "3"
EOF

    kubectl apply -f /tmp/configmap-temp.yaml
    rm /tmp/configmap-temp.yaml
    print_status "ConfigMap updated successfully"
    
    # Create empty knowledge-base ConfigMap
    create_knowledge_base_configmap
}

# Function to update secrets
update_secrets() {
    local litellm_api_key="$1"
    local embedding_api_key="$2"
    local tavily_api_key="$3"
    local langfuse_public_key="$4"
    local langfuse_secret_key="$5"
    
    print_status "Updating Secrets with provided API keys..."
    
    # Create temporary secrets file matching the existing structure
    cat > /tmp/secrets-temp.yaml << EOF
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  labels:
    app: strandsdk-rag
type: Opaque
data:
  # Base64 encoded secrets - replace with your actual base64 encoded values
  # To encode: echo -n "your-api-key" | base64
  litellm-api-key: $(encode_base64 "$litellm_api_key")
  embedding-api-key: $(encode_base64 "$embedding_api_key")
  tavily-api-key: $(encode_base64 "$tavily_api_key")
  langfuse-public-key: $(encode_base64 "$langfuse_public_key")
  langfuse-secret-key: $(encode_base64 "$langfuse_secret_key")
EOF

    kubectl apply -f /tmp/secrets-temp.yaml
    rm /tmp/secrets-temp.yaml
    print_status "Secrets updated successfully"
}

# Main setup function
main() {
    print_header "Multi-Agent RAG System Setup"
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists kubectl; then
        print_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    if ! command_exists aws; then
        print_error "AWS CLI is not installed. Please install AWS CLI first."
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if we can connect to Kubernetes cluster
    if ! kubectl cluster-info >/dev/null 2>&1; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_status "Prerequisites check passed!"
    
    # Step 1: Build and push Docker images
    print_header "Step 1: Building and Pushing Docker Images"
    
    if [ -f "./build-images.sh" ]; then
        print_status "Running build-images.sh..."
        chmod +x ./build-images.sh
        ./build-images.sh
        print_status "Docker images built and pushed successfully!"
    else
        print_warning "build-images.sh not found. Skipping image build step."
        echo "Please ensure your container images are available in ECR."
        read -p "Press Enter to continue..."
    fi
    
    # Step 2: Deploy OpenSearch
    print_header "Step 2: Deploying OpenSearch Cluster"
    
    get_input "Enter OpenSearch stack name" "strandsdk-rag-opensearch-stack" STACK_NAME
    get_input "Enter AWS region" "us-east-1" AWS_REGION
    get_input "Enter Kubernetes namespace" "default" NAMESPACE
    
    if [ -f "./deploy-opensearch.sh" ]; then
        print_status "Running deploy-opensearch.sh..."
        chmod +x ./deploy-opensearch.sh
        ./deploy-opensearch.sh "$STACK_NAME" "$AWS_REGION" "$NAMESPACE"
        print_status "OpenSearch cluster deployed successfully!"
        
        # Get OpenSearch endpoint from CloudFormation stack
        print_status "Retrieving OpenSearch endpoint..."
        OPENSEARCH_ENDPOINT=$(aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --region "$AWS_REGION" \
            --query 'Stacks[0].Outputs[?OutputKey==`OpenSearchEndpoint`].OutputValue' \
            --output text 2>/dev/null || echo "")
        
        if [ -n "$OPENSEARCH_ENDPOINT" ]; then
            print_status "OpenSearch endpoint: $OPENSEARCH_ENDPOINT"
        else
            print_warning "Could not retrieve OpenSearch endpoint automatically."
            get_input "Please enter the OpenSearch endpoint manually" "" OPENSEARCH_ENDPOINT
        fi
    else
        print_warning "deploy-opensearch.sh not found. Skipping OpenSearch deployment."
        get_input "Please enter your existing OpenSearch endpoint" "" OPENSEARCH_ENDPOINT
    fi
    
    # Step 3: Configure application settings
    print_header "Step 3: Configuring Application Settings"
    
    # Auto-detect LiteLLM endpoint
    print_status "Auto-detecting service endpoints..."
    LITELLM_DEFAULT=""
    if LITELLM_DETECTED=$(get_litellm_ingress); then
        LITELLM_DEFAULT="$LITELLM_DETECTED"
        log_litellm_detection
        print_status "Auto-detected LiteLLM endpoint: $LITELLM_DEFAULT"
    else
        print_warning "Could not auto-detect LiteLLM endpoint"
        LITELLM_DEFAULT="https://api.openai.com/v1"
    fi
    
    # Auto-detect Langfuse endpoint
    LANGFUSE_DEFAULT=""
    if LANGFUSE_DETECTED=$(get_langfuse_ingress); then
        LANGFUSE_DEFAULT="$LANGFUSE_DETECTED"
        log_langfuse_detection
        print_status "Auto-detected Langfuse endpoint: $LANGFUSE_DEFAULT"
    else
        print_warning "Could not auto-detect Langfuse endpoint"
        LANGFUSE_DEFAULT="https://cloud.langfuse.com"
    fi
    
    # Get user input with auto-detected defaults
    get_input "Enter LiteLLM base URL" "$LITELLM_DEFAULT" LITELLM_BASE_URL
    
    # For embedding, use the same base URL but with /embeddings endpoint
    EMBEDDING_DEFAULT="${LITELLM_BASE_URL%/v1}/v1/embeddings"
    get_input "Enter Embedding service base URL" "$EMBEDDING_DEFAULT" EMBEDDING_BASE_URL
    
    get_input "Enter Langfuse host" "$LANGFUSE_DEFAULT" LANGFUSE_HOST
    
    # Step 4: Configure secrets
    print_header "Step 4: Configuring API Keys and Secrets"
    
    print_status "Please provide the following API keys:"
    
    # Required API keys
    get_secret_input "Enter LiteLLM API key" LITELLM_API_KEY
    get_secret_input "Enter Embedding service API key (can be same as LiteLLM)" EMBEDDING_API_KEY
    get_secret_input "Enter Tavily API key (for web search functionality)" TAVILY_API_KEY
    
    echo
    print_status "Optional Langfuse keys for observability (press Enter to skip):"
    read -p "Enter Langfuse public key (optional): " LANGFUSE_PUBLIC_KEY
    read -p "Enter Langfuse secret key (optional): " LANGFUSE_SECRET_KEY
    
    # Set defaults for empty Langfuse keys
    LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-""}
    LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-""}
    
    # Validate required fields
    if [ -z "$LITELLM_API_KEY" ] || [ -z "$EMBEDDING_API_KEY" ] || [ -z "$TAVILY_API_KEY" ]; then
        print_error "Required API keys are missing. Please provide all required keys."
        exit 1
    fi
    
    # Step 5: Apply Kubernetes resources
    print_header "Step 5: Applying Kubernetes Resources"
    
    # Update ConfigMap
    update_configmap "$OPENSEARCH_ENDPOINT" "$LITELLM_BASE_URL" "$EMBEDDING_BASE_URL" "$LANGFUSE_HOST"
    
    # Update Secrets
    update_secrets "$LITELLM_API_KEY" "$EMBEDDING_API_KEY" "$TAVILY_API_KEY" "$LANGFUSE_PUBLIC_KEY" "$LANGFUSE_SECRET_KEY"
    
    # Apply remaining Kubernetes resources
    print_status "Applying Kubernetes manifests..."
    
    if [ -f "k8s/service-account.yaml" ]; then
        kubectl apply -f k8s/service-account.yaml
        print_status "Service account applied"
    else
        print_warning "k8s/service-account.yaml not found"
    fi
    
    if [ -f "k8s/tavily-mcp-deployment.yaml" ]; then
        kubectl apply -f k8s/tavily-mcp-deployment.yaml
        print_status "Tavily MCP deployment applied"
    else
        print_warning "k8s/tavily-mcp-deployment.yaml not found"
    fi
    
    if [ -f "k8s/main-app-deployment.yaml" ]; then
        kubectl apply -f k8s/main-app-deployment.yaml
        print_status "Main application deployment applied"
    else
        print_warning "k8s/main-app-deployment.yaml not found"
    fi
    
    # Step 6: Verify deployment
    print_header "Step 6: Verifying Deployment"
    
    print_status "Checking pod status..."
    kubectl get pods -l app=tavily-mcp-server
    kubectl get pods -l app=strandsdk-rag-app
    
    print_status "Checking services..."
    kubectl get svc
    
    print_status "Checking ingress..."
    kubectl get ingress
    
    # Get final ALB endpoint
    print_status "Waiting for ALB endpoint to be ready..."
    sleep 30
    
    ALB_ENDPOINT=$(kubectl get ingress strandsdk-rag-ingress-alb -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    if [ -n "$ALB_ENDPOINT" ]; then
        print_header "Deployment Complete!"
        echo -e "${GREEN}Your Multi-Agent RAG system is now deployed!${NC}"
        echo
        echo -e "${BLUE}Configuration Summary:${NC}"
        echo "• OpenSearch Endpoint: $OPENSEARCH_ENDPOINT"
        echo "• LiteLLM Base URL: $LITELLM_BASE_URL"
        echo "• Embedding Base URL: $EMBEDDING_BASE_URL"
        echo "• Langfuse Host: $LANGFUSE_HOST"
        echo "• Application Load Balancer: http://$ALB_ENDPOINT"
        echo
        echo -e "${BLUE}Test endpoints:${NC}"
        echo "Health check: curl -X GET \"http://$ALB_ENDPOINT/health\""
        echo "Embed knowledge: curl -X POST \"http://$ALB_ENDPOINT/embed\" -H \"Content-Type: application/json\" -d '{\"force_refresh\": false}'"
        echo "Complex query: curl -i -X POST \"http://$ALB_ENDPOINT/query\" -H \"Content-Type: application/json\" -d '{\"question\": \"Find information about \\\"What was the purpose of the study on encainide and flecainide in patients with supraventricular arrhythmias\\\". Summarize this information and create a comprehensive story.Save the story and important information to a file named \\\"test1.md\\\" in the output directory as a beautiful markdown file.\", \"top_k\": 3}' --max-time 600"
        echo
        echo -e "${YELLOW}Note: It may take a few minutes for the ALB to become fully available.${NC}"
        echo -e "${YELLOW}If services were auto-detected, verify the endpoints are correct for your setup.${NC}"
    else
        print_warning "ALB endpoint not yet available. Check 'kubectl get ingress' in a few minutes."
        echo
        echo -e "${BLUE}Configuration Summary:${NC}"
        echo "• OpenSearch Endpoint: $OPENSEARCH_ENDPOINT"
        echo "• LiteLLM Base URL: $LITELLM_BASE_URL"
        echo "• Embedding Base URL: $EMBEDDING_BASE_URL"
        echo "• Langfuse Host: $LANGFUSE_HOST"
    fi
    
    print_status "Setup completed successfully!"
}

# Run main function
main "$@"
