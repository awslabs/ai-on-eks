# SGLang Deployments

This directory contains SGLang deployment configurations for the NVIDIA Dynamo platform with RadixAttention caching.

## Available Deployments

| Deployment | Description | Model | Resources |
|------------|-------------|-------|-----------|
| `sglang-aggregated-default` | Single worker with RadixAttention | DeepSeek-R1-Distill-Llama-8B | 1 GPU, 10 CPU, 20Gi RAM |
| `sglang-disaggregated-default` | Separate prefill/decode workers | DeepSeek-R1-Distill-Llama-8B | 1+1 GPUs, 8 CPU each |
| `sglang-router` | KV-aware routing for cache optimization | Configurable | Configurable |

## Architecture

### Aggregated Architecture (`sglang-aggregated-default`)
- **Single worker** handles both prefill and decode phases
- **RadixAttention caching** for efficient memory management
- **Better for**: Single-user scenarios, simpler deployment

### Disaggregated Architecture (`sglang-disaggregated-default`)
- **Separate workers** for prefill and decode phases with NIXL transfer backend
- **Better for**: High throughput, concurrent requests, production workloads
- **Communication**: Uses NIXL (NVIDIA Inter-X Link) for worker coordination

## Key Features

### SGLang Optimizations
- **RadixAttention**: Advanced attention mechanism with automatic prefix caching
- **Prefix Sharing**: Automatic detection and reuse of common prompt prefixes
- **Memory Efficiency**: Up to 5x reduction in memory usage for repetitive queries
- **Fast Sampling**: Optimized token generation algorithms
- **Dynamic Batching**: Efficient request batching and scheduling
- **NIXL Transfer Backend**: High-speed inter-worker communication for disaggregated mode

### Integration Benefits
- **Automatic Model Discovery**: Workers register automatically with frontend
- **Advanced Caching**: RadixAttention provides intelligent cache management
- **OpenAI Compatible API**: Standard `/v1/chat/completions` endpoints
- **Namespace Management**: Automatic namespace clearing on startup

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- G5 GPU nodes available (at least 1-2 GPUs with 24GB VRAM each)
- HuggingFace token secret configured

## Quick Start

### Deploy Aggregated SGLang
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh sglang-aggregated-default
```

### Deploy Disaggregated SGLang
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh sglang-disaggregated-default
```

## Configuration Details

### Aggregated Default
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Resources**: 1 GPU, 10 CPU, 20Gi RAM
- **Features**: RadixAttention with 16-page size, trust remote code, skip tokenizer init
- **Frontend**: Higher resource allocation (5 CPU, 10Gi) due to namespace clearing

### Disaggregated Default
- **Model**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Architecture**: SGLangPrefillWorker + SGLangDecodeWorker
- **Resources**: 1 GPU, 8 CPU, 20Gi RAM per worker
- **Communication**: NIXL transfer backend for high-speed worker coordination
- **Features**: Same RadixAttention optimizations across both workers

## SGLang-Specific Parameters

### Common Parameters
```bash
--model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
--served-model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
--page-size 16                    # RadixAttention page size
--tp 1                           # Tensor parallelism
--trust-remote-code              # Allow custom model code
--skip-tokenizer-init            # Faster startup
```

### Disaggregated-Specific Parameters
```bash
--disaggregation-mode prefill     # For prefill worker
--disaggregation-mode decode      # For decode worker
--disaggregation-transfer-backend nixl  # High-speed communication
```

## Testing

### Basic Health Check
```bash
# Port forward to frontend service
kubectl port-forward service/sglang-aggregated-default-frontend 8000:8000 -n dynamo-cloud

# Test health endpoint
curl http://localhost:8000/health
```

### Chat Completions
```bash
# Test reasoning capabilities with DeepSeek model
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
# Test prefix caching with repeated prompts
for i in {1..3}; do
  echo "Request $i:"
  time curl -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant. Always be concise."},
        {"role": "user", "content": "What is machine learning?"}
      ],
      "max_tokens": 50
    }' 2>/dev/null | jq '.choices[0].message.content'
done
# Subsequent requests should be faster due to prefix caching
```

## Monitoring

### Pod Status
```bash
# Check deployment status
kubectl get dynamographdeployment sglang-aggregated-default -n dynamo-cloud

# Check all SGLang pods
kubectl get pods -n dynamo-cloud -l app=sglang-aggregated-default
```

### Logs and RadixAttention Metrics
```bash
# Frontend logs (includes namespace clearing)
kubectl logs -n dynamo-cloud -l componentType=main,app=sglang-aggregated-default -f

# Worker logs (includes RadixAttention cache activity)
kubectl logs -n dynamo-cloud -l componentType=worker,app=sglang-aggregated-default -f

# Check for cache hits and RadixAttention activity
kubectl logs -n dynamo-cloud -l componentType=worker -f | grep -i "cache\|radix\|hit"
```

### Disaggregated Monitoring
```bash
# Check both prefill and decode workers
kubectl get pods -n dynamo-cloud -l app=sglang-disaggregated-default

# Prefill worker logs
kubectl logs -n dynamo-cloud -l app=sglang-disaggregated-default | grep prefill

# Decode worker logs
kubectl logs -n dynamo-cloud -l app=sglang-disaggregated-default | grep decode
```

## GPU Requirements and Node Selection

### Default Node Configuration
```yaml
nodeSelector:
  karpenter.sh/nodepool: g5-gpu-karpenter
tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

### Why G5 for SGLang
- **A10G Memory Bandwidth**: Sufficient for RadixAttention operations
- **Cost Effectiveness**: Best price/performance for cache-heavy workloads
- **RadixAttention Optimization**: Good balance of compute and memory for caching

### Alternative Configurations

**For Cache-Heavy Workloads:**
```yaml
nodeSelector:
  karpenter.sh/nodepool: g6-gpu-karpenter  # L4 GPUs with higher bandwidth
```

## Performance Tuning

### RadixAttention Optimization
```yaml
args:
  - "--page-size"
  - "32"          # Larger pages for longer contexts (default: 16)
  - "--max-total-tokens"
  - "32768"       # Maximum context length
```

### For High Throughput
- Use disaggregated architecture
- Scale workers horizontally
- Monitor NIXL transfer performance
- Optimize page size for workload

### For Low Latency
- Use aggregated architecture
- Enable aggressive caching
- Use consistent prompt patterns to maximize prefix reuse

## External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments.

**SGLang-Specific Notes:**
- RadixAttention benefits from session affinity for multi-turn conversations
- Consider enabling sticky sessions for optimal cache performance

## Cleanup

```bash
# Remove deployment
kubectl delete dynamographdeployment sglang-aggregated-default -n dynamo-cloud
# or
kubectl delete dynamographdeployment sglang-disaggregated-default -n dynamo-cloud
```

## Troubleshooting

### Common Issues

**Namespace Clearing Issues:**
```bash
# Check if namespace clearing completed
kubectl logs -n dynamo-cloud -l componentType=main -f | grep "clear_namespace"
```

**Worker Registration Issues:**
```bash
# Check worker registration in logs
kubectl logs -n dynamo-cloud -l componentType=worker -f | grep -i "register\|ready"
```

**RadixAttention Performance:**
```bash
# Monitor cache performance
kubectl logs -n dynamo-cloud -l componentType=worker -f | grep -i "cache\|hit\|miss"

# Check page size configuration
kubectl logs -n dynamo-cloud -l componentType=worker -f | grep -i "page-size"
```

## References

- [SGLang Official Documentation](https://github.com/sgl-project/sglang)
- [RadixAttention Paper](https://arxiv.org/abs/2312.07104)
- [NVIDIA Dynamo Documentation](https://docs.nvidia.com/dynamo/)
- [DeepSeek-R1 Model Documentation](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
