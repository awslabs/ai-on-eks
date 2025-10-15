# vLLM Deployments

This directory contains vLLM deployment configurations for the NVIDIA Dynamo platform.

## Available Deployments

| Deployment | Description | Model | Resources |
|------------|-------------|-------|-----------|
| `vllm-aggregated-default` | Single worker with tensor parallelism | Qwen/Qwen3-8B | 2 GPUs, 10 CPU, 20Gi RAM |
| `vllm-disaggregated-default` | Separate prefill/decode workers | Qwen/Qwen3-0.6B | 1+1 GPUs, 8 CPU each |
| `vllm-router` | KV-aware routing for cache optimization | Configurable | Configurable |

## Architecture

### Aggregated Architecture (`vllm-aggregated-default`)
- **Single worker** handles both prefill and decode phases
- **Tensor parallelism** across multiple GPUs for better performance
- **Better for**: Single-user scenarios, lower latency

### Disaggregated Architecture (`vllm-disaggregated-default`)
- **Separate workers** for prefill and decode phases
- **Better for**: High throughput, concurrent requests, production workloads
- **Resource usage**: GPUs split between prefill and decode workers

## Key Features

### vLLM Optimizations
- **Continuous Batching**: Dynamic request batching for maximum throughput
- **PagedAttention**: Memory-efficient attention computation
- **Quantization Support**: GPTQ, AWQ, and SqueezeLLM support
- **Tensor Parallelism**: Multi-GPU support for large models
- **OpenAI Compatible API**: Standard `/v1/chat/completions` endpoints

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- G5 GPU nodes available (at least 1-2 GPUs with 24GB VRAM each)
- HuggingFace token secret configured

## Quick Start

### Deploy Aggregated vLLM
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh vllm-aggregated-default
```

### Deploy Disaggregated vLLM
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh vllm-disaggregated-default
```

## Configuration Details

### Aggregated Default
- **Model**: `Qwen/Qwen3-8B` (8B parameter model)
- **GPUs**: 2 GPUs with `--tensor-parallel-size 2`
- **Resources**: 10 CPU, 20Gi RAM per worker
- **Node type**: G5 GPU instances (`g5-gpu-karpenter`)

### Disaggregated Default
- **Model**: `Qwen/Qwen3-0.6B` (smaller, faster model)
- **Architecture**: Separate prefill and decode workers
- **Resources**: 1 GPU, 8 CPU, 20Gi RAM per worker
- **Workers**: VllmPrefillWorker + VllmDecodeWorker

## Testing

### Basic Health Check
```bash
# Port forward to frontend service
kubectl port-forward service/vllm-aggregated-default-frontend 8000:8000 -n dynamo-cloud

# Test health endpoint
curl http://localhost:8000/health
```

### Chat Completions
```bash
# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "What is artificial intelligence?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

### Model Discovery
```bash
# List available models
curl http://localhost:8000/v1/models
```

## Monitoring

### Pod Status
```bash
# Check deployment status
kubectl get dynamographdeployment vllm-aggregated-default -n dynamo-cloud

# Check pods
kubectl get pods -n dynamo-cloud -l app=vllm-aggregated-default
```

### Logs
```bash
# Frontend logs
kubectl logs -n dynamo-cloud -l componentType=main,app=vllm-aggregated-default -f

# Worker logs
kubectl logs -n dynamo-cloud -l componentType=worker,app=vllm-aggregated-default -f
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

### Recommended Instance Types
- **G5.2xlarge**: 1x A10G GPU (24GB) - for disaggregated workers
- **G5.4xlarge**: 1x A10G GPU (24GB) - for aggregated single GPU
- **G5.12xlarge**: 4x A10G GPU (96GB total) - for aggregated tensor parallelism

## External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments.

## Cleanup

```bash
# Remove deployment
kubectl delete dynamographdeployment vllm-aggregated-default -n dynamo-cloud
# or
kubectl delete dynamographdeployment vllm-disaggregated-default -n dynamo-cloud
```

## Troubleshooting

### Common Issues

**Model Download Issues:**
```bash
# Check HuggingFace token secret
kubectl get secret hf-token-secret -n dynamo-cloud

# Check worker logs for download progress
kubectl logs -n dynamo-cloud -l componentType=worker -f
```

**GPU Resource Issues:**
```bash
# Check GPU availability
kubectl describe nodes -l karpenter.sh/nodepool=g5-gpu-karpenter

# Check resource requests vs limits
kubectl describe pod <pod-name> -n dynamo-cloud
```

## References

- [vLLM Documentation](https://vllm.readthedocs.io/)
- [NVIDIA Dynamo Documentation](https://docs.nvidia.com/dynamo/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
