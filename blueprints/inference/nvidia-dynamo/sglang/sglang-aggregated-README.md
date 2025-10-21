# SGLang Example

Deploy SGLang-based LLM serving with advanced RadixAttention caching using NVIDIA Dynamo v0.5.0.

## Architecture

```text
Client Requests → Frontend → SGLang Worker (Aggregated + RadixAttention)
```

This example demonstrates:
- SGLang backend integration with RadixAttention caching
- Advanced memory management for high-throughput serving
- Multi-model discovery and aggregation
- OpenAI-compatible API with enhanced caching performance
- G5 GPU node selection for cost-effective deployment

## Key Features

### RadixAttention Caching
SGLang's RadixAttention provides significant performance improvements:
- **Prefix Caching**: Automatic detection and reuse of common prompt prefixes
- **KV Cache Optimization**: Intelligent cache eviction and management
- **Memory Efficiency**: Up to 5x reduction in memory usage for repetitive queries
- **Latency Reduction**: 2-10x faster response times for cached prefixes

### SGLang-Specific Optimizations
- **Advanced Batching**: Dynamic batching for better throughput
- **Memory Pooling**: Efficient GPU memory management
- **Structured Generation**: Built-in support for JSON/XML output formats
- **Fast Tokenization**: Optimized tokenizer with caching

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- G5 GPU nodes available (at least 1 GPU with 24GB VRAM)
- HuggingFace token with model access permissions

## YAML Structure Explained

### Frontend Configuration
```yaml
Frontend:
  dynamoNamespace: sglang          # Service discovery namespace
  componentType: main              # HTTP API entry point
  replicas: 1                      # Single frontend instance
  resources:
    requests:
      cpu: "5"                     # Higher CPU for request processing
      memory: "10Gi"               # Memory for routing and caching metadata
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: cpu-karpenter  # CPU-only node
    mainContainer:
      image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.5.0
      workingDir: /workspace/components/backends/sglang
      args:
        # Clear namespace for clean startup
        - "python3 -m dynamo.sglang.utils.clear_namespace --namespace sglang && python3 -m dynamo.frontend --http-port=8000"
```

**Key Points:**
- **Namespace Clearing**: SGLang clears its namespace on startup to prevent stale worker registrations
- **Higher Resource Allocation**: SGLang frontend requires more CPU/memory than other backends
- **CPU Node Placement**: Frontend doesn't need GPU, runs on cost-effective CPU nodes

### Worker Configuration
```yaml
SGLangWorker:
  dynamoNamespace: sglang          # Must match frontend namespace
  componentType: worker            # Inference processing unit
  envFromSecret: hf-token-secret   # HuggingFace authentication
  replicas: 1                      # Single worker (can scale)
  resources:
    requests:
      cpu: "10"                    # High CPU for model operations
      memory: "20Gi"               # Large memory for model + cache
      gpu: "1"                     # Single GPU requirement
  extraPodSpec:
    nodeSelector:
      karpenter.sh/nodepool: g5-gpu-karpenter  # G5 GPU instances
    tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
    mainContainer:
      image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.5.0
      workingDir: /workspace/components/backends/sglang
      args:
        - "python3"
        - "-m"
        - "dynamo.sglang.worker"
        - "--model-path"
        - "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Large, capable model
        - "--served-model-name"
        - "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        - "--page-size"
        - "16"                     # RadixAttention page size (tune for workload)
        - "--tp"
        - "1"                      # Tensor parallelism (1 GPU)
        - "--trust-remote-code"    # Enable custom model code
        - "--skip-tokenizer-init"  # Optimize startup time
```

**Key Parameters:**
- **Model Selection**: Uses DeepSeek-R1 model for advanced reasoning capabilities
- **Page Size**: RadixAttention cache page size (16 is balanced for most workloads)
- **Trust Remote Code**: Enables custom model implementations
- **Skip Tokenizer Init**: Faster worker startup by deferring tokenizer initialization

## Node Selection Strategy

### GPU Worker Placement
```yaml
extraPodSpec:
  nodeSelector:
    karpenter.sh/nodepool: g5-gpu-karpenter  # G5 instances for cost-effectiveness
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

**Why G5 for SGLang:**
- **Memory Bandwidth**: A10G GPUs provide sufficient bandwidth for RadixAttention
- **Cost Effectiveness**: Best price/performance for cache-heavy workloads
- **Availability**: Good availability across regions
- **Model Size**: 24GB VRAM handles most models up to 20B parameters

### Alternative Node Configurations

**For Cache-Heavy Workloads:**
```yaml
nodeSelector:
  karpenter.sh/nodepool: g6-gpu-karpenter  # L4 GPUs with higher memory bandwidth
```

**For Large Models:**
```yaml
nodeSelector:
  karpenter.sh/nodepool: p5-gpu-karpenter
  node.kubernetes.io/instance-type: p5.48xlarge  # H100 for 70B+ models
```

## Deployment

### Using the Deployment Script (Recommended)
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh sglang
```

### Manual Deployment
```bash
# Ensure HuggingFace token secret exists
kubectl get secret hf-token-secret -n dynamo-cloud

# Deploy SGLang
kubectl apply -f sglang/sglang.yaml -n dynamo-cloud

# Monitor deployment
kubectl get pods -n dynamo-cloud -l app=sglang -w
```

## Model Configuration

This example uses `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` which demonstrates SGLang's capabilities well. You can modify the deployment YAML to use other supported models like:

- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-0.6B` (for smaller resource requirements)
- `mistralai/Mistral-7B-Instruct-v0.3`

## Testing

### Basic Health Check
```bash
# Port forward to frontend
# Port forward via Service (recommended) - enables both API access and metrics collection
kubectl port-forward service/sglang-frontend 8000:8000 -n dynamo-cloud

# Alternative: Direct deployment access
# kubectl port-forward deployment/sglang-frontend 8000:8000 -n dynamo-cloud

# Test health endpoint
curl http://localhost:8000/health
# Expected: {"endpoints":["dyn://dynamo.worker.generate","dyn://dynamo.backend.generate"],"status":"healthy"}
```

### Model Discovery
```bash
# Check available models (includes cross-backend discovery)
curl http://localhost:8000/v1/models
# Expected: List of available models from all discovered workers
```

### Chat Completions
```bash
# Test reasoning capabilities
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}
    ],
    "max_tokens": 200,
    "temperature": 0.1
  }'
```

### Cache Performance Test
```bash
# Test prefix caching with repeated prompt
for i in {1..3}; do
  echo "Request $i:"
  time curl -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant. Always start your response with a greeting."},
        {"role": "user", "content": "What is the capital of France?"}
      ],
      "max_tokens": 50
    }' 2>/dev/null | jq '.choices[0].message.content'
done
# Subsequent requests should be significantly faster due to prefix caching
```

## Monitoring

### Pod Status
```bash
# Check all SGLang components
kubectl get pods -n dynamo-cloud -l app=sglang

# Check worker specifically
kubectl get pods -n dynamo-cloud -l componentType=worker,app=sglang
```

### Logs and Debugging
```bash
# Frontend logs
kubectl logs -n dynamo-cloud -l componentType=main,app=sglang -f

# Worker logs
kubectl logs -n dynamo-cloud -l componentType=worker,app=sglang -f

# Check for cache hits in worker logs
kubectl logs -n dynamo-cloud -l componentType=worker,app=sglang | grep -i "cache\|hit\|radix"
```

### Performance Metrics
```bash
# Check DynamoGraphDeployment status
kubectl get dynamographdeployment sglang -n dynamo-cloud -o yaml

# Monitor resource usage
kubectl top pods -n dynamo-cloud -l app=sglang
```

## Advanced Configuration

### Tuning RadixAttention
```yaml
args:
  - "--page-size"
  - "32"          # Larger pages for longer contexts (default: 16)
  - "--max-total-tokens"
  - "32768"       # Maximum context length
  - "--disable-disk-cache"  # Disable disk caching for pure memory operation
```

### Multi-GPU Setup
```yaml
resources:
  requests:
    gpu: "2"      # Request 2 GPUs
extraPodSpec:
  nodeSelector:
    node.kubernetes.io/instance-type: g5.12xlarge  # 4 GPU instance
args:
  - "--tp"
  - "2"          # 2-way tensor parallelism
```

### Custom Health Probes for Large Models
```yaml
readinessProbe:
  exec:
    command:
      - /bin/sh
      - -c
      - 'python3 -c "import requests; requests.get(\"http://localhost:9090/health\").raise_for_status()"'
  initialDelaySeconds: 180    # 3 minutes for large model loading
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 20        # Allow extended startup time
```

## Troubleshooting

### Common Issues

**Worker Not Starting:**
```bash
# Check GPU availability
kubectl describe node <gpu-node> | grep -i gpu

# Check model download progress
kubectl logs <sglang-worker-pod> -n dynamo-cloud | grep -i download
```

**Frontend Not Finding Worker:**
```bash
# Verify namespace clearing worked
kubectl logs <sglang-frontend-pod> -n dynamo-cloud | grep -i "clear_namespace"

# Check worker registration
kubectl logs <sglang-worker-pod> -n dynamo-cloud | grep -i "register\|ready"
```

**Poor Cache Performance:**
```bash
# Check RadixAttention initialization
kubectl logs <sglang-worker-pod> -n dynamo-cloud | grep -i "radix\|cache"

# Verify page size settings
kubectl logs <sglang-worker-pod> -n dynamo-cloud | grep -i "page-size"
```

### Performance Optimization

**For High Throughput:**
- Increase `--page-size` to 32 or 64
- Use G6 instances for higher memory bandwidth
- Scale workers horizontally

**For Low Latency:**
- Use smaller models (e.g., Qwen3-0.6B)
- Enable aggressive caching with larger cache sizes
- Use on-demand instances for consistent performance

## External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments.

**SGLang-Specific Notes:**
- RadixAttention caching benefits from session affinity (sticky sessions)
- Consider enabling ALB sticky sessions for multi-turn conversations
- See root README.md for complete setup instructions

## Cleanup

```bash
# Remove SGLang deployment
kubectl delete dynamographdeployment sglang -n dynamo-cloud

# Verify cleanup
kubectl get pods -n dynamo-cloud -l app=sglang
```

## References

- [SGLang Official Documentation](https://github.com/sgl-project/sglang)
- [RadixAttention Paper](https://arxiv.org/abs/2312.07104)
- [NVIDIA Dynamo Architecture Guide](https://docs.nvidia.com/dynamo/)
- [DeepSeek-R1 Model Documentation](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
