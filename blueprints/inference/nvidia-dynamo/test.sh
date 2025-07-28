#!/bin/bash

#---------------------------------------------------------------
# NVIDIA Dynamo Inference Graph Testing
#
# Simple testing script for deployed inference graphs.
# Tests health, metrics, and completions endpoints.
#
# Usage:
#   ./test.sh [service_name]
#
# Examples:
#   ./test.sh                        # Interactive selection
#   ./test.sh llm-disagg-router-frontend
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

print_banner "DYNAMO INFERENCE GRAPH TESTING"

# Load environment configuration
if [ -f "${ENV_FILE}" ]; then
    source "${ENV_FILE}"
    info "Loaded environment configuration from ${ENV_FILE}"
else
    error "Environment file not found: ${ENV_FILE}"
    error "Please run the infrastructure installation first: infra/nvidia-dynamo/install.sh"
    exit 1
fi

# Update kubeconfig
info "Updating kubeconfig..."
aws eks update-kubeconfig --region ${AWS_REGION} --name ${CLUSTER_NAME}

#---------------------------------------------------------------
# Service Discovery
#---------------------------------------------------------------

section "Service Discovery"

# Get available services
AVAILABLE_SERVICES=($(kubectl get services -n ${NAMESPACE} -o name | cut -d'/' -f2))

if [ ${#AVAILABLE_SERVICES[@]} -eq 0 ]; then
    error "No services found in namespace ${NAMESPACE}"
    exit 1
fi

# Select service
SERVICE_NAME=""

if [ $# -gt 0 ]; then
    SERVICE_NAME="$1"
    # Validate provided service
    if [[ ! " ${AVAILABLE_SERVICES[@]} " =~ " ${SERVICE_NAME} " ]]; then
        error "Service '${SERVICE_NAME}' not found in namespace ${NAMESPACE}"
        info "Available services: ${AVAILABLE_SERVICES[*]}"
        exit 1
    fi
else
    # Interactive selection
    info "Available services:"
    for i in "${!AVAILABLE_SERVICES[@]}"; do
        echo "  $((i+1)). ${AVAILABLE_SERVICES[i]}"
    done

    echo -n "Select a service (1-${#AVAILABLE_SERVICES[@]}): "
    read -r selection

    if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#AVAILABLE_SERVICES[@]} ]; then
        SERVICE_NAME="${AVAILABLE_SERVICES[$((selection-1))]}"
    else
        error "Invalid selection"
        exit 1
    fi
fi

info "Selected service: ${SERVICE_NAME}"

#---------------------------------------------------------------
# Service Information
#---------------------------------------------------------------

section "Service Information"

# Get service details
SERVICE_PORT=$(kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].port}')
SERVICE_TYPE=$(kubectl get service ${SERVICE_NAME} -n ${NAMESPACE} -o jsonpath='{.spec.type}')
LOCAL_PORT=${SERVICE_PORT:-8000}

info "Service: ${SERVICE_NAME}"
info "Port: ${SERVICE_PORT}"
info "Type: ${SERVICE_TYPE}"

# Check if service is ready
READY_REPLICAS=$(kubectl get deployment -n ${NAMESPACE} -l app=${SERVICE_NAME} -o jsonpath='{.items[0].status.readyReplicas}' 2>/dev/null || echo "0")
DESIRED_REPLICAS=$(kubectl get deployment -n ${NAMESPACE} -l app=${SERVICE_NAME} -o jsonpath='{.items[0].status.replicas}' 2>/dev/null || echo "1")

info "Ready replicas: ${READY_REPLICAS}/${DESIRED_REPLICAS}"

if [ "${READY_REPLICAS}" != "${DESIRED_REPLICAS}" ]; then
    warn "Service may not be fully ready. Continuing with tests..."
fi

#---------------------------------------------------------------
# Port Forwarding Setup
#---------------------------------------------------------------

section "Port Forwarding Setup"

# Start port forwarding in background
info "Setting up port forwarding to localhost:${LOCAL_PORT}..."
kubectl port-forward service/${SERVICE_NAME} ${LOCAL_PORT}:${SERVICE_PORT} -n ${NAMESPACE} &
PORT_FORWARD_PID=$!

# Wait for port forwarding to be ready
sleep 5

# Function to cleanup port forwarding
cleanup() {
    if [ -n "${PORT_FORWARD_PID:-}" ]; then
        info "Cleaning up port forwarding..."
        kill ${PORT_FORWARD_PID} 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

#---------------------------------------------------------------
# Health Check
#---------------------------------------------------------------

section "Health Check"

# Test basic connectivity
info "Testing basic connectivity..."
HEALTH_URL="http://localhost:${LOCAL_PORT}/health"

# Try health endpoint first
if curl -s -f "${HEALTH_URL}" >/dev/null 2>&1; then
    success "Health endpoint is accessible"
    curl -s "${HEALTH_URL}" | jq . 2>/dev/null || curl -s "${HEALTH_URL}"
else
    warn "Health endpoint not accessible, trying root endpoint..."
    ROOT_URL="http://localhost:${LOCAL_PORT}/"

    if curl -s -f "${ROOT_URL}" >/dev/null 2>&1; then
        success "Root endpoint is accessible"
        curl -s "${ROOT_URL}" | head -10
    else
        error "Service is not responding on port ${LOCAL_PORT}"
        info "Please check if the service is running:"
        echo "  kubectl get pods -n ${NAMESPACE}"
        echo "  kubectl logs -n ${NAMESPACE} -l app=${SERVICE_NAME}"
        exit 1
    fi
fi

#---------------------------------------------------------------
# API Testing
#---------------------------------------------------------------

section "API Testing"

# Test key endpoints
BASE_URL="http://localhost:${LOCAL_PORT}"

# Test endpoints that worked in the original output
ENDPOINTS=(
    "/health"
    "/metrics"
    "/v1/models"
)

info "Testing key endpoints..."
for endpoint in "${ENDPOINTS[@]}"; do
    url="${BASE_URL}${endpoint}"

    if curl -s -f "${url}" >/dev/null 2>&1; then
        success "✓ ${endpoint} - accessible"
    else
        info "✗ ${endpoint} - not accessible"
    fi
done

#---------------------------------------------------------------
# Sample Inference Tests
#---------------------------------------------------------------

section "Chat Completions Test"

# Test chat completions endpoint (the one that worked in your output)
COMPLETIONS_URL="${BASE_URL}/v1/chat/completions"

info "Testing chat completions endpoint..."
# Use the correct model name from the Dynamo LLM examples
CHAT_PAYLOAD='{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
        {"role": "user", "content": "Explain Quantum computing in relation to classical computing. Connect it to digital vs analog computing. Use the latest research and examples."}
    ],
    "max_tokens": 500,
    "temperature": 0.7
}'

RESPONSE=$(curl -s -X POST "${COMPLETIONS_URL}" \
    -H "Content-Type: application/json" \
    -d "${CHAT_PAYLOAD}" 2>/dev/null || echo "")

if [ -n "${RESPONSE}" ]; then
    success "Chat completions endpoint responded"
    echo "Response:"
    echo "${RESPONSE}" | jq . 2>/dev/null || echo "${RESPONSE}"

    # Check if we got a "Model not found" error and try alternative model names
    if echo "${RESPONSE}" | grep -q "Model not found"; then
        warn "Model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' not found, trying alternative model names..."

        # Try with just the model name without the org prefix
        ALT_PAYLOAD='{
            "model": "DeepSeek-R1-Distill-Llama-8B",
            "messages": [
                {"role": "user", "content": "Explain Quantum computing in relation to classical computing. Connect it to digital vs analog computing. Use the latest research and examples."}
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }'

        ALT_RESPONSE=$(curl -s -X POST "${COMPLETIONS_URL}" \
            -H "Content-Type: application/json" \
            -d "${ALT_PAYLOAD}" 2>/dev/null || echo "")

        if [ -n "${ALT_RESPONSE}" ] && ! echo "${ALT_RESPONSE}" | grep -q "Model not found"; then
            success "Alternative model name worked!"
            echo "Response:"
            echo "${ALT_RESPONSE}" | jq . 2>/dev/null || echo "${ALT_RESPONSE}"
        else
            # Try with "default" model name
            DEFAULT_PAYLOAD='{
                "model": "default",
                "messages": [
                    {"role": "user", "content": "Explain Quantum computing in relation to classical computing. Connect it to digital vs analog computing. Use the latest research and examples."}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }'

            DEFAULT_RESPONSE=$(curl -s -X POST "${COMPLETIONS_URL}" \
                -H "Content-Type: application/json" \
                -d "${DEFAULT_PAYLOAD}" 2>/dev/null || echo "")

            if [ -n "${DEFAULT_RESPONSE}" ] && ! echo "${DEFAULT_RESPONSE}" | grep -q "Model not found"; then
                success "Default model name worked!"
                echo "Response:"
                echo "${DEFAULT_RESPONSE}" | jq . 2>/dev/null || echo "${DEFAULT_RESPONSE}"
            else
                warn "All model names failed. Check deployment configuration or model availability."
            fi
        fi
    fi
else
    warn "Chat completions endpoint did not respond"
fi

#---------------------------------------------------------------
# Performance Test
#---------------------------------------------------------------

section "Performance Test"

info "Running enhanced performance test..."

# Test 1: Health endpoint performance (quick baseline)
info "Testing health endpoint performance (3 requests)..."
HEALTH_TIMES=()
for i in {1..3}; do
    RESPONSE_TIME=$(curl -s -w "%{time_total}" -o /dev/null "${BASE_URL}/health" 2>/dev/null || echo "timeout")
    HEALTH_TIMES+=("$RESPONSE_TIME")
    echo "Health request $i: ${RESPONSE_TIME}s"
done

# Test 2: Chat completions performance (more challenging)
info "Testing chat completions performance (3 requests)..."

# Use a simpler prompt for performance testing to reduce variability
PERF_PAYLOAD='{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
        {"role": "user", "content": "What is artificial intelligence?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}'

COMPLETION_TIMES=()
for i in {1..3}; do
    echo "Sending completion request $i..."
    START_TIME=$(date +%s.%N)

    RESPONSE=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "${PERF_PAYLOAD}" 2>/dev/null || echo "")

    END_TIME=$(date +%s.%N)
    RESPONSE_TIME=$(echo "${END_TIME} - ${START_TIME}" | bc -l 2>/dev/null || echo "0")
    COMPLETION_TIMES+=("$RESPONSE_TIME")

    if [ -n "${RESPONSE}" ] && ! echo "${RESPONSE}" | grep -q "error"; then
        echo "Completion request $i: ${RESPONSE_TIME}s ✓"
    else
        echo "Completion request $i: ${RESPONSE_TIME}s ✗ (error)"
    fi
done

# Calculate and display averages
if command -v bc >/dev/null 2>&1; then
    HEALTH_AVG=$(printf '%s\n' "${HEALTH_TIMES[@]}" | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "0"}')
    COMPLETION_AVG=$(printf '%s\n' "${COMPLETION_TIMES[@]}" | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "0"}')

    echo ""
    success "Performance test completed"
    echo "Average response times:"
    echo "  Health endpoint: ${HEALTH_AVG}s"
    echo "  Chat completions: ${COMPLETION_AVG}s"
    echo "  Performance ratio: $(echo "scale=1; ${COMPLETION_AVG} / ${HEALTH_AVG}" | bc -l 2>/dev/null || echo "N/A")x slower"
else
    success "Performance test completed (install 'bc' for detailed statistics)"
fi

#---------------------------------------------------------------
# Summary
#---------------------------------------------------------------

section "Test Summary"

success "Testing completed for service: ${SERVICE_NAME}"

echo ""
echo "Service Information:"
echo "  Name: ${SERVICE_NAME}"
echo "  Namespace: ${NAMESPACE}"
echo "  Port: ${SERVICE_PORT}"
echo "  Local URL: http://localhost:${LOCAL_PORT}"
echo ""

echo "Manual Testing Commands:"
echo "  1. Port forwarding: kubectl port-forward service/${SERVICE_NAME} ${LOCAL_PORT}:${SERVICE_PORT} -n ${NAMESPACE}"
echo "  2. Health check: curl http://localhost:${LOCAL_PORT}/health"
echo "  3. Chat completions: curl -X POST http://localhost:${LOCAL_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"deepseek-ai/DeepSeek-R1-Distill-Llama-8B\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 50}'"
echo "  4. View logs: kubectl logs -n ${NAMESPACE} -l app=${SERVICE_NAME}"
echo "  5. Monitor pods: kubectl get pods -n ${NAMESPACE} -w"
