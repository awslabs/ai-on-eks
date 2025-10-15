# vLLM Example

Deploy vLLM-based LLM serving with aggregated architecture using Dynamo v0.5.0.

## Architecture

```text
Client Requests → Frontend → vLLM Worker (Aggregated)
```

This example demonstrates:
- vLLM backend integration with NVIDIA Dynamo
- OpenAI-compatible API serving (`/v1/chat/completions`, `/v1/models`)
- Aggregated serving mode (prefill + decode in same worker)
- G5 GPU node selection for cost-effective inference
- Production-ready health checks and resource management

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- G5 GPU nodes available (at least 1 GPU with 24GB VRAM)
- HuggingFace token secret configured

## YAML Structure Explained

### Frontend Configuration
```yaml
Frontend:
  dynamoNamespace: vllm             # Service discovery namespace
  componentType: main               # HTTP API entry point
  replicas: 1                       # Single frontend instance
  resources:
    requests:
      cpu: "1"                      # Lightweight for request routing
      memory: "2Gi"                 # Minimal memory for HTTP server
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: cpu-karpenter  # CPU-only node (cost effective)
    mainContainer:
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
      workingDir: /workspace/components/backends/vllm
      args: ["python3", "-m", "dynamo.frontend", "--http-port", "8000"]
  livenessProbe:
    httpGet:
      path: /health
      port: 8000
  readinessProbe:
    exec:
      command: ["/bin/sh", "-c", 'curl -s http://localhost:8000/health | jq -e ".status == \"healthy\""']
```

**Key Points:**
- **OpenAI API**: Provides standard `/v1/chat/completions` endpoint
- **Service Discovery**: Automatically finds vLLM workers in same namespace
- **Health Checks**: Comprehensive HTTP and shell-based probes
- **CPU Placement**: Frontend doesn't need GPU, runs on cheaper CPU nodes

### Worker Configuration
```yaml
VllmWorker:
  dynamoNamespace: vllm             # Must match frontend namespace
  componentType: worker             # Inference processing unit
  envFromSecret: hf-token-secret    # HuggingFace authentication
  replicas: 1                       # Single worker (can scale horizontally)
  resources:
    requests:
      gpu: "1"                      # Single GPU requirement
      cpu: "10"                     # High CPU for model operations
      memory: "20Gi"                # Large memory for model + KV cache
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: g5-gpu-karpenter  # G5 GPU instances
    tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
    mainContainer:
      image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.5.0
      workingDir: /workspace/components/backends/vllm
      args: ["python3", "-m", "dynamo.vllm", "--model", "Qwen/Qwen3-0.6B", "2>&1", "|", "tee", "/tmp/vllm.log"]
  envs:
    - name: DYN_SYSTEM_ENABLED
      value: "true"                 # Enable Dynamo system integration
    - name: DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS
      value: "[\"generate\"]"       # Health check endpoint
    - name: DYN_SYSTEM_PORT
      value: "9090"                 # Internal health port
```

**Key Parameters:**
- **Model Loading**: Uses `Qwen/Qwen3-0.6B` (small, fast model for testing)
- **Resource Allocation**: Balanced CPU/memory for aggregated serving
- **Health Integration**: Dynamo system handles service health reporting
- **GPU Scheduling**: Automatically scheduled on G5 GPU nodes

## Node Selection Strategy

### GPU Worker Placement
```yaml
extraPodSpec:
  nodeSelector:
    karpenter.sh/nodepool: g5-gpu-karpenter
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

**Why G5 for vLLM:**
- **Memory Capacity**: 24GB VRAM handles models up to ~20B parameters
- **Price/Performance**: Best cost efficiency for development and production
- **Availability**: Good regional availability, reliable provisioning
- **vLLM Optimization**: A10G GPUs well-supported by vLLM

### Alternative Configurations

**For Large Models (8B+ parameters):**
```yaml
nodeSelector:
  karpenter.sh/nodepool: g6-gpu-karpenter  # L4 GPUs with higher bandwidth
```

**For Multi-GPU Tensor Parallelism:**
```yaml
nodeSelector:
  karpenter.sh/nodepool: g5-gpu-karpenter
  node.kubernetes.io/instance-type: g5.12xlarge  # 4 GPU instance
resources:
  requests:
    gpu: "2"                        # Request multiple GPUs
args:
  - "--tensor-parallel-size"
  - "2"                             # Enable tensor parallelism
```

## Deployment

Deploy using kubectl:

```bash
kubectl apply -f vllm.yaml -n dynamo-cloud
```

Or use the main deployment script:

```bash
cd ..
./deploy.sh vllm
```

## Model Configuration

This example uses `Qwen/Qwen3-0.6B` which is a small model suitable for testing. For production workloads, you can modify the deployment YAML to use larger models like:

- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

## Testing

Once deployed, test the vLLM service:

```bash
# Port-forward the frontend via Service (created automatically by deploy script)
# Port forward via Service (recommended) - enables both API access and metrics collection
kubectl port-forward service/vllm-frontend 8000:8000 -n dynamo-cloud

# Alternative: Direct deployment access
# kubectl port-forward deployment/vllm-frontend 8000:8000 -n dynamo-cloud

# Test health endpoint
curl http://localhost:8000/health

# Test chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'

# List models
curl http://localhost:8000/v1/models
```

### External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments, including:
- AWS Load Balancer Controller setup
- Ingress configurations
- Best practices for target-type and performance optimization


## Monitoring

Check the deployment status:

```bash
# View pods
kubectl get pods -n dynamo-cloud -l app=vllm

# Check logs
kubectl logs -n dynamo-cloud -l app=vllm-worker -f

# View DynamoGraphDeployment status
kubectl get dynamographdeployment vllm -n dynamo-cloud -o yaml
```

## Cleanup

Remove the deployment:

```bash
kubectl delete dynamographdeployment vllm -n dynamo-cloud
```
