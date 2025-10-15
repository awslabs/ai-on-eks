#!/bin/bash

#---------------------------------------------------------------
# NVIDIA Dynamo v0.4.1 Example Testing Script
#
# Simple testing script for deployed Dynamo examples.
# Tests health, metrics, and API endpoints based on example type.
#
# Usage:
#   ./test.sh [example-name]
#
# Examples:
#   ./test.sh hello-world   # Test hello-world example
#   ./test.sh vllm         # Test vLLM deployment
#   ./test.sh sglang       # Test SGLang deployment
#   ./test.sh trtllm       # Test TensorRT-LLM deployment
#   ./test.sh multinode-vllm # Test multi-node vLLM deployment
#   ./test.sh              # Interactive selection
#---------------------------------------------------------------

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default namespace
NAMESPACE="dynamo-cloud"

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

print_banner "DYNAMO EXAMPLE TESTING"

#---------------------------------------------------------------
# Example Selection
#---------------------------------------------------------------

section "Example Selection"

# Get available examples dynamically from deploy.sh
get_available_examples() {
    local deploy_script="${SCRIPT_DIR}/deploy.sh"
    if [ -f "$deploy_script" ]; then
        # Extract examples from deploy.sh AVAILABLE_EXAMPLES array
        grep -A 20 "AVAILABLE_EXAMPLES=(" "$deploy_script" | \
        grep -E '^\s*"[^"]+:[^"]*"' | \
        sed 's/.*"\([^:]*\):.*/\1/' | \
        sort
    else
        # Fallback to common examples if deploy.sh not found
        echo "hello-world vllm sglang trtllm-default trtllm-high-performance multi-replica-vllm vllm-disagg sglang-disagg trtllm-disagg-default trtllm-disagg-high-performance kv-routing"
    fi
}

AVAILABLE_EXAMPLES=($(get_available_examples))

EXAMPLE=""
if [ $# -gt 0 ]; then
    EXAMPLE="$1"
    # Validate provided example against available examples
    if [[ ! " ${AVAILABLE_EXAMPLES[@]} " =~ " ${EXAMPLE} " ]]; then
        error "Invalid example: ${EXAMPLE}"
        info "Available examples: ${AVAILABLE_EXAMPLES[*]}"
        exit 1
    fi
else
    # Check for deployed examples dynamically
    info "Checking for deployed examples..."
    DEPLOYED_EXAMPLES=()

    # Get all deployed DynamoGraphDeployments
    if kubectl get dynamographdeployments -n "${NAMESPACE}" >/dev/null 2>&1; then
        while IFS= read -r deployment_name; do
            if [ -n "$deployment_name" ] && [ "$deployment_name" != "NAME" ]; then
                DEPLOYED_EXAMPLES+=("$deployment_name")
            fi
        done < <(kubectl get dynamographdeployments -n "${NAMESPACE}" --no-headers -o custom-columns=":metadata.name" 2>/dev/null)
    fi

    if [ ${#DEPLOYED_EXAMPLES[@]} -eq 0 ]; then
        error "No deployed examples found in namespace ${NAMESPACE}"
        info "Available examples to deploy: ${AVAILABLE_EXAMPLES[*]}"
        info "Deploy an example first: ./deploy.sh <example-name>"
        exit 1
    fi

    if [ ${#DEPLOYED_EXAMPLES[@]} -eq 1 ]; then
        EXAMPLE="${DEPLOYED_EXAMPLES[0]}"
        info "Found deployed example: ${EXAMPLE}"
    else
        info "Multiple deployed examples found:"
        for i in "${!DEPLOYED_EXAMPLES[@]}"; do
            echo "  $((i+1)). ${DEPLOYED_EXAMPLES[i]}"
        done
        echo ""

        while true; do
            read -p "Select an example to test (1-${#DEPLOYED_EXAMPLES[@]}): " selection
            if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le ${#DEPLOYED_EXAMPLES[@]} ]; then
                EXAMPLE="${DEPLOYED_EXAMPLES[$((selection-1))]}"
                break
            else
                error "Invalid selection. Please choose 1-${#DEPLOYED_EXAMPLES[@]}."
            fi
        done
    fi
fi

info "Testing example: ${EXAMPLE}"

#---------------------------------------------------------------
# Prerequisites Check
#---------------------------------------------------------------

section "Prerequisites Check"

# Check if example is deployed
if ! kubectl get dynamographdeployment "$EXAMPLE" -n "${NAMESPACE}" >/dev/null 2>&1; then
    error "Example '${EXAMPLE}' is not deployed in namespace '${NAMESPACE}'"
    info "Deploy it first: ./deploy.sh ${EXAMPLE}"
    exit 1
fi
success "Example '${EXAMPLE}' is deployed"

# Check if service exists
SERVICE_NAME="${EXAMPLE}-frontend"
if ! kubectl get service "$SERVICE_NAME" -n "${NAMESPACE}" >/dev/null 2>&1; then
    warn "Frontend service '${SERVICE_NAME}' not found, checking for alternative names..."
    # Try some common alternatives
    for alt in "${EXAMPLE}" "${EXAMPLE}-app" "${EXAMPLE}-svc"; do
        if kubectl get service "$alt" -n "${NAMESPACE}" >/dev/null 2>&1; then
            SERVICE_NAME="$alt"
            success "Found service: ${SERVICE_NAME}"
            break
        fi
    done

    if [[ "$SERVICE_NAME" == "${EXAMPLE}-frontend" ]]; then
        error "No suitable service found for example '${EXAMPLE}'"
        info "Available services in namespace ${NAMESPACE}:"
        kubectl get services -n "${NAMESPACE}" | grep "$EXAMPLE" || echo "  (none found)"
        exit 1
    fi
else
    success "Frontend service found: ${SERVICE_NAME}"
fi

#---------------------------------------------------------------
# Service Information
#---------------------------------------------------------------

section "Service Information"

# Get service details
SERVICE_PORT=$(kubectl get service "$SERVICE_NAME" -n "${NAMESPACE}" -o jsonpath='{.spec.ports[0].port}' 2>/dev/null || echo "8000")
SERVICE_TYPE=$(kubectl get service "$SERVICE_NAME" -n "${NAMESPACE}" -o jsonpath='{.spec.type}' 2>/dev/null || echo "ClusterIP")

# Find available local port
find_available_port() {
    local start_port=${1:-8000}
    local max_port=$((start_port + 100))

    for ((port=start_port; port<=max_port; port++)); do
        if ! ss -tuln | grep -q ":${port} "; then
            echo "$port"
            return 0
        fi
    done

    # Fallback to a high port if nothing found
    echo "9000"
}

LOCAL_PORT=$(find_available_port ${SERVICE_PORT:-8000})

info "Service: ${SERVICE_NAME}"
info "Port: ${SERVICE_PORT}"
info "Type: ${SERVICE_TYPE}"
info "Local port for testing: ${LOCAL_PORT}"

# Check pod status
info ""
info "Pod status:"
kubectl get pods -n "${NAMESPACE}" -l "app=${EXAMPLE}" 2>/dev/null || {
    warn "No pods found with label app=${EXAMPLE}"
}

#---------------------------------------------------------------
# Port Forwarding Setup
#---------------------------------------------------------------

section "Port Forwarding Setup"

# Start port forwarding in background
info "Setting up port forwarding to localhost:${LOCAL_PORT}..."

# Clean up any existing port forwards for this service
pkill -f "port-forward.*${SERVICE_NAME}" 2>/dev/null || true

kubectl port-forward service/"$SERVICE_NAME" "$LOCAL_PORT:$SERVICE_PORT" -n "$NAMESPACE" &
PORT_FORWARD_PID=$!

# Verify port forwarding started successfully
sleep 2
if ! kill -0 $PORT_FORWARD_PID 2>/dev/null; then
    error "Port forwarding failed to start"
    # Try with a different port
    LOCAL_PORT=$(find_available_port $((LOCAL_PORT + 1)))
    warn "Retrying with port ${LOCAL_PORT}..."
    kubectl port-forward service/"$SERVICE_NAME" "$LOCAL_PORT:$SERVICE_PORT" -n "$NAMESPACE" &
    PORT_FORWARD_PID=$!
    sleep 2
fi

# Function to cleanup port forwarding
cleanup() {
    if [ -n "${PORT_FORWARD_PID:-}" ]; then
        info "Cleaning up port forwarding..."
        kill ${PORT_FORWARD_PID} 2>/dev/null || true
        # Also kill any lingering port-forward processes for this service
        pkill -f "port-forward.*${SERVICE_NAME}" 2>/dev/null || true
    fi
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Wait for port forwarding to be ready
info "Waiting for port forwarding to be ready..."
sleep 5

#---------------------------------------------------------------
# Basic Health Check
#---------------------------------------------------------------

section "Health Check"

BASE_URL="http://localhost:${LOCAL_PORT}"

# Test basic connectivity
info "Testing basic connectivity..."
HEALTH_URL="${BASE_URL}/health"

if curl -s -f "$HEALTH_URL" >/dev/null 2>&1; then
    success "Health endpoint is accessible"
    HEALTH_RESPONSE=$(curl -s "$HEALTH_URL")
    echo "Health response:"
    echo "$HEALTH_RESPONSE" | jq . 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    warn "Health endpoint not accessible, trying root endpoint..."
    ROOT_URL="${BASE_URL}/"

    if curl -s -f "$ROOT_URL" >/dev/null 2>&1; then
        success "Root endpoint is accessible"
        curl -s "$ROOT_URL" | head -10
    else
        error "Service is not responding on port ${LOCAL_PORT}"
        info "Please check if the service is running:"
        echo "  kubectl get pods -n ${NAMESPACE} -l app=${EXAMPLE}"
        echo "  kubectl logs -n ${NAMESPACE} -l app=${EXAMPLE}"
        exit 1
    fi
fi

#---------------------------------------------------------------
# Example-Specific Testing
#---------------------------------------------------------------

section "Example-Specific Tests"

case "$EXAMPLE" in
    "hello-world")
        info "Testing hello-world specific endpoints..."
        # Test any hello-world specific endpoints
        ;;

    "vllm-aggregated-default"|"vllm-disaggregated-default"|"sglang-aggregated-default"|"sglang-disaggregated-default"|"trtllm-aggregated-default"|"trtllm-aggregated-high-performance"|"trtllm-disaggregated-default"|"multi-replica-vllm"|"vllm-router"|"sglang-router"|"trtllm-router")
        info "Testing LLM service endpoints..."

        # Test models endpoint
        MODELS_URL="${BASE_URL}/v1/models"
        if curl -s -f "$MODELS_URL" >/dev/null 2>&1; then
            success "✓ /v1/models - accessible"
            echo "Available models:"
            curl -s "$MODELS_URL" | jq '.data[].id' 2>/dev/null || curl -s "$MODELS_URL"
        else
            warn "✗ /v1/models - not accessible"
        fi

        echo ""

        # Test chat completions with a simple request
        info "Testing chat completions endpoint..."
        COMPLETIONS_URL="${BASE_URL}/v1/chat/completions"

        # Dynamic model selection
        info "Discovering available models..."
        MODELS_RESPONSE=$(curl -s "$MODELS_URL" 2>/dev/null || echo "")

        if [ -n "$MODELS_RESPONSE" ] && ! echo "$MODELS_RESPONSE" | grep -q -i "error"; then
            # Extract model names from the response
            AVAILABLE_MODELS=()
            if command -v jq >/dev/null 2>&1; then
                # Try to parse as JSON with .data array first (OpenAI format)
                if echo "$MODELS_RESPONSE" | jq -e '.data' >/dev/null 2>&1; then
                    while IFS= read -r model; do
                        if [ -n "$model" ] && [ "$model" != "null" ]; then
                            AVAILABLE_MODELS+=("$model")
                        fi
                    done < <(echo "$MODELS_RESPONSE" | jq -r '.data[]?.id // empty' 2>/dev/null)
                # Try to parse as simple JSON array or string
                elif echo "$MODELS_RESPONSE" | jq -e '.' >/dev/null 2>&1; then
                    # Check if it's a simple string (quoted model name)
                    if echo "$MODELS_RESPONSE" | jq -e '. | type == "string"' >/dev/null 2>&1; then
                        MODEL_NAME=$(echo "$MODELS_RESPONSE" | jq -r '.')
                        AVAILABLE_MODELS+=("$MODEL_NAME")
                    # Check if it's an array of strings
                    elif echo "$MODELS_RESPONSE" | jq -e '. | type == "array"' >/dev/null 2>&1; then
                        while IFS= read -r model; do
                            if [ -n "$model" ] && [ "$model" != "null" ]; then
                                AVAILABLE_MODELS+=("$model")
                            fi
                        done < <(echo "$MODELS_RESPONSE" | jq -r '.[]' 2>/dev/null)
                    fi
                fi
            fi

            # Fallback: extract from plain text if jq parsing failed
            if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
                # Try to extract quoted strings (model names)
                while IFS= read -r line; do
                    if [[ "$line" =~ \"([^\"]+)\" ]]; then
                        model="${BASH_REMATCH[1]}"
                        if [[ "$model" != "data" ]] && [[ "$model" != "id" ]] && [[ "$model" != "object" ]]; then
                            AVAILABLE_MODELS+=("$model")
                        fi
                    fi
                done <<< "$MODELS_RESPONSE"
            fi

            # Model selection logic
            if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
                warn "No models found in response, falling back to generic model name"
                MODEL_NAME="default-model"
            elif [ ${#AVAILABLE_MODELS[@]} -eq 1 ]; then
                MODEL_NAME="${AVAILABLE_MODELS[0]}"
                info "Using the only available model: ${MODEL_NAME}"
            else
                info "Multiple models available:"
                for i in "${!AVAILABLE_MODELS[@]}"; do
                    echo "  $((i+1)). ${AVAILABLE_MODELS[i]}"
                done
                echo ""

                # Interactive model selection
                while true; do
                    read -p "Select a model for testing (1-${#AVAILABLE_MODELS[@]}) or press Enter for first model: " model_selection

                    if [ -z "$model_selection" ]; then
                        # Default to first model if user just presses Enter
                        MODEL_NAME="${AVAILABLE_MODELS[0]}"
                        info "Using default model: ${MODEL_NAME}"
                        break
                    elif [[ "$model_selection" =~ ^[0-9]+$ ]] && [ "$model_selection" -ge 1 ] && [ "$model_selection" -le ${#AVAILABLE_MODELS[@]} ]; then
                        MODEL_NAME="${AVAILABLE_MODELS[$((model_selection-1))]}"
                        info "Using selected model: ${MODEL_NAME}"
                        break
                    else
                        error "Invalid selection. Please choose 1-${#AVAILABLE_MODELS[@]} or press Enter for default."
                    fi
                done
            fi
        else
            warn "Could not retrieve models list, falling back to default model selection"
            # Fallback to example-based model names as backup
            case "$EXAMPLE" in
                "vllm-aggregated-default") MODEL_NAME="Qwen/Qwen3-8B" ;;
                "vllm-disaggregated-default"|"multi-replica-vllm"|"vllm-router") MODEL_NAME="Qwen/Qwen3-0.6B" ;;
                "sglang-aggregated-default"|"sglang-disaggregated-default"|"sglang-router") MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ;;
                "trtllm-aggregated-default"|"trtllm-aggregated-high-performance"|"trtllm-disaggregated-default"|"trtllm-router") MODEL_NAME="Qwen/Qwen3-0.6B" ;;
                *) MODEL_NAME="default-model" ;;
            esac
            info "Using fallback model: ${MODEL_NAME}"
        fi

        CHAT_PAYLOAD=$(cat <<EOF
{
    "model": "${MODEL_NAME}",
    "messages": [
        {"role": "user", "content": "What is quantum computing and what are its parallels to analogue and digital computing? Explain Like I am 5 years old."}
    ],
    "max_tokens": 500,
    "temperature": 0.1
}
EOF
)

        echo "Testing with model: ${MODEL_NAME}"
        RESPONSE=$(curl -s -X POST "$COMPLETIONS_URL" \
            -H "Content-Type: application/json" \
            -d "$CHAT_PAYLOAD" 2>/dev/null || echo "")

        if [ -n "$RESPONSE" ] && ! echo "$RESPONSE" | grep -q -i "error"; then
            success "✓ Chat completions endpoint responded successfully"
            echo "Response preview:"
            echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "$RESPONSE" | head -3
        else
            warn "✗ Chat completions endpoint failed or returned error"
            if [ -n "$RESPONSE" ]; then
                echo "Error response:"
                echo "$RESPONSE" | head -5

                # Check for common instance ID routing issues
                if echo "$RESPONSE" | grep -q "instance_id.*not found"; then
                    warn "Detected instance ID routing issue - this may indicate:"
                    echo "  1. Frontend has cached old instance IDs from a previous deployment"
                    echo "  2. Workers are still starting up or failed to register properly"
                    echo "  3. Network connectivity issues between frontend and workers"
                    echo ""
                    echo "To fix this issue:"
                    echo "  1. Wait for all worker pods to be fully ready: kubectl get pods -n ${NAMESPACE} -l app=${EXAMPLE}"
                    echo "  2. Check worker logs: kubectl logs -n ${NAMESPACE} -l app=${EXAMPLE},component=*Worker"
                    echo "  3. Restart frontend pod to clear cache: kubectl delete pod -n ${NAMESPACE} -l app=${EXAMPLE},component=Frontend"
                fi
            fi
        fi

        # Advanced testing for specific examples
        case "$EXAMPLE" in
            "vllm-disaggregated-default"|"sglang-disaggregated-default"|"trtllm-disaggregated-default")
                echo ""
                info "Testing disaggregation with long context..."
                LONG_CONTEXT=$(python3 -c "print('Long context test: ' + 'word ' * 100)")
                LONG_PAYLOAD=$(cat <<EOF
{
    "model": "${MODEL_NAME}",
    "messages": [
        {"role": "user", "content": "${LONG_CONTEXT}. Summarize this."}
    ],
    "max_tokens": 50
}
EOF
)
                LONG_RESPONSE=$(curl -s -X POST "$COMPLETIONS_URL" \
                    -H "Content-Type: application/json" \
                    -d "$LONG_PAYLOAD" 2>/dev/null || echo "")

                if [ -n "$LONG_RESPONSE" ] && ! echo "$LONG_RESPONSE" | grep -q -i "error"; then
                    success "✓ Long context request (disaggregation test) succeeded"
                else
                    warn "✗ Long context request failed (check disaggregation setup)"
                fi
                ;;

            "vllm-router"|"sglang-router"|"trtllm-router")
                echo ""
                info "Testing KV routing with shared prefixes..."
                SHARED_SYSTEM="You are a helpful AI assistant."

                # Clean up any existing test files
                rm -f /tmp/kv_test_*.json 2>/dev/null

                # Store background job PIDs
                KV_PIDS=()

                for i in {1..3}; do
                    KV_PAYLOAD=$(cat <<EOF
{
    "model": "${MODEL_NAME}",
    "messages": [
        {"role": "system", "content": "${SHARED_SYSTEM}"},
        {"role": "user", "content": "Question ${i}: What is AI?"}
    ],
    "max_tokens": 30
}
EOF
)
                    # Add timeout to curl command and run in background
                    (
                        curl -s -m 30 -X POST "$COMPLETIONS_URL" \
                            -H "Content-Type: application/json" \
                            -d "$KV_PAYLOAD" > /tmp/kv_test_$i.json 2>/dev/null || \
                        echo "timeout_or_error" > /tmp/kv_test_$i.json
                    ) &
                    KV_PIDS+=($!)
                done

                # Wait for all requests with timeout
                info "Waiting for KV routing test requests (max 45 seconds)..."
                WAIT_COUNT=0
                MAX_WAIT=45

                while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
                    # Check if all jobs are done
                    JOBS_RUNNING=false
                    for pid in "${KV_PIDS[@]}"; do
                        if kill -0 "$pid" 2>/dev/null; then
                            JOBS_RUNNING=true
                            break
                        fi
                    done

                    if [ "$JOBS_RUNNING" = false ]; then
                        break
                    fi

                    sleep 1
                    WAIT_COUNT=$((WAIT_COUNT + 1))
                done

                # Kill any remaining jobs if timeout reached
                if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
                    warn "KV routing test timed out, killing remaining requests..."
                    for pid in "${KV_PIDS[@]}"; do
                        kill "$pid" 2>/dev/null || true
                    done
                fi

                # Count successful responses
                success_count=0
                error_count=0

                for i in {1..3}; do
                    if [ -f "/tmp/kv_test_$i.json" ]; then
                        if ! grep -q "timeout_or_error" "/tmp/kv_test_$i.json" 2>/dev/null; then
                            success_count=$((success_count + 1))
                        else
                            error_count=$((error_count + 1))
                        fi
                    else
                        error_count=$((error_count + 1))
                    fi
                done

                if [ $success_count -eq 3 ]; then
                    success "✓ KV routing test: ${success_count}/3 requests completed successfully"
                elif [ $success_count -gt 0 ]; then
                    warn "✓ KV routing test: ${success_count}/3 requests completed (${error_count} failed/timed out)"
                else
                    warn "✗ KV routing test: All requests failed or timed out"
                fi

                # Clean up test files
                rm -f /tmp/kv_test_*.json 2>/dev/null
                ;;
        esac
        ;;

    "sla-planner")
        info "Testing SLA planner specific endpoints..."

        # Check if Prometheus is available
        PROMETHEUS_URL="${BASE_URL}:9090"
        if curl -s -f "${PROMETHEUS_URL}/-/healthy" >/dev/null 2>&1; then
            success "✓ Prometheus endpoint accessible for SLA planner"
        else
            warn "✗ Prometheus not accessible (may affect SLA planner metrics)"
        fi

        # Test the main LLM endpoint
        MODELS_URL="${BASE_URL}/v1/models"
        if curl -s -f "$MODELS_URL" >/dev/null 2>&1; then
            success "✓ LLM service accessible for SLA monitoring"
        else
            warn "✗ LLM service not yet ready (check planner initialization)"
        fi
        ;;
esac

#---------------------------------------------------------------
# Performance Test
#---------------------------------------------------------------

section "Performance Test"

info "Running basic performance test..."

# Test health endpoint response time
info "Testing health endpoint performance (3 requests)..."
HEALTH_TIMES=()
for i in {1..3}; do
    RESPONSE_TIME=$(curl -s -w "%{time_total}" -o /dev/null "$BASE_URL/health" 2>/dev/null || echo "timeout")
    HEALTH_TIMES+=("$RESPONSE_TIME")
    echo "Health request $i: ${RESPONSE_TIME}s"
done

# Calculate average if bc is available
if command -v bc >/dev/null 2>&1; then
    HEALTH_AVG=$(printf '%s\n' "${HEALTH_TIMES[@]}" | awk '{sum+=$1; count++} END {if(count>0) printf "%.3f", sum/count; else print "0"}')
    echo "Average health response time: ${HEALTH_AVG}s"
fi

#---------------------------------------------------------------
# Summary
#---------------------------------------------------------------

section "Test Summary"

success "Testing completed for example: ${EXAMPLE}"

echo ""
echo "Service Information:"
echo "  Example: ${EXAMPLE}"
echo "  Service: ${SERVICE_NAME}"
echo "  Namespace: ${NAMESPACE}"
echo "  Port: ${SERVICE_PORT}"
echo "  Local URL: http://localhost:${LOCAL_PORT}"
echo ""

echo "Manual Testing Commands:"
echo "  1. Port forwarding: kubectl port-forward service/${SERVICE_NAME} ${LOCAL_PORT}:${SERVICE_PORT} -n ${NAMESPACE}"
echo "  2. Health check: curl http://localhost:${LOCAL_PORT}/health"

case "$EXAMPLE" in
    "vllm-aggregated-default"|"vllm-disaggregated-default"|"sglang-aggregated-default"|"sglang-disaggregated-default"|"trtllm-aggregated-default"|"trtllm-aggregated-high-performance"|"trtllm-disaggregated-default"|"multi-replica-vllm"|"vllm-router"|"sglang-router"|"trtllm-router")
        echo "  3. List models: curl http://localhost:${LOCAL_PORT}/v1/models"
        echo "  4. Chat completion: curl -X POST http://localhost:${LOCAL_PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"${MODEL_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}], \"max_tokens\": 50}'"
        ;;
esac

echo "  5. View logs: kubectl logs -n ${NAMESPACE} -l app=${EXAMPLE}"
echo "  6. Monitor pods: kubectl get pods -n ${NAMESPACE} -l app=${EXAMPLE} -w"
echo ""

echo "Cleanup:"
echo "  kubectl delete dynamographdeployment ${EXAMPLE} -n ${NAMESPACE}"
