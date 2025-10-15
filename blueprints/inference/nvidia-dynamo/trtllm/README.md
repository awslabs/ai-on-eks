# TensorRT-LLM Deployments

This directory contains TensorRT-LLM deployment configurations for the NVIDIA Dynamo platform with maximum inference performance optimization.

## Available Deployments

| Deployment | Description | Model | Resources |
|------------|-------------|-------|-----------|
| `trtllm-aggregated-default` | Single worker with default settings | Qwen/Qwen3-0.6B | 1 GPU, 10 CPU, 20Gi RAM |
| `trtllm-aggregated-high-performance` | Optimized for maximum throughput | Qwen/Qwen3-0.6B | 1 GPU, 16 CPU, 32Gi RAM |
| `trtllm-disaggregated-default` | Separate prefill/decode workers | Qwen/Qwen3-0.6B | 1+1 GPUs, 8 CPU each |
| `trtllm-router` | KV-aware routing for cache optimization | Configurable | Configurable |

## Architecture

### Aggregated Architecture
- **Single worker** with embedded engine configurations
- **Better for**: Lower latency, simpler deployment
- **Resource usage**: All compute in one pod

### Disaggregated Architecture
- **Separate prefill and decode workers** with specialized configurations
- **Better for**: High throughput, concurrent requests, production workloads
- **Resource usage**: GPUs split between prefill and decode workers
- **Communication**: Cache transceiver for worker coordination

## Key Features

### TensorRT-LLM Optimizations
- **Custom CUDA Kernels**: Hand-optimized kernels for maximum GPU utilization
- **Memory Optimization**: Advanced KV cache management and memory pooling
- **Chunked Prefill**: Efficient handling of long sequences
- **CUDA Graphs**: Reduced kernel launch overhead for consistent performance
- **Inline Configuration**: Embedded engine configurations (no external ConfigMaps)
- **Cache Transceiver**: Optimized communication for disaggregated setups

### NGC Authentication Required
⚠️ **All TensorRT-LLM deployments require NGC (NVIDIA GPU Cloud) authentication** to pull container images.

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- **NGC API Key secret configured** (`ngc-secret`)
- G5 GPU nodes available (at least 1-2 GPUs with 24GB VRAM each)
- HuggingFace token secret configured

## Quick Start

### Deploy Default TensorRT-LLM
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh trtllm-aggregated-default
```

### Deploy High-Performance TensorRT-LLM
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh trtllm-aggregated-high-performance
```

### Deploy Disaggregated TensorRT-LLM
```bash
cd blueprints/inference/nvidia-dynamo
./deploy.sh trtllm-disaggregated-default
```

## Configuration Details

### Default Configuration
- **Model**: `Qwen/Qwen3-0.6B` (small, fast model)
- **Resources**: 1 GPU, 10 CPU, 20Gi RAM
- **Batch Size**: 16 (conservative for stability)
- **Max Tokens**: 8192
- **Memory Usage**: 85% GPU memory for KV cache
- **Use Case**: Balanced performance for development and testing

### High-Performance Configuration
- **Model**: `Qwen/Qwen3-0.6B`
- **Resources**: 1 GPU, 16 CPU, 32Gi RAM (higher resources)
- **Batch Size**: 32 (aggressive batching)
- **Max Tokens**: 16384
- **Memory Usage**: 90% GPU memory for KV cache (maximum utilization)
- **Use Case**: Maximum throughput for production workloads

### Disaggregated Configuration
- **Model**: `Qwen/Qwen3-0.6B`
- **Architecture**: TRTLLMPrefillWorker + TRTLLMDecodeWorker
- **Resources**: 1 GPU, 8 CPU, 20Gi RAM per worker
- **Strategy**: `decode_first` for optimal request handling
- **Cache Transceiver**: Default backend for worker communication
- **Use Case**: High-throughput, concurrent request processing

## Inline Engine Configurations

All TensorRT-LLM deployments use embedded engine configurations created at runtime:

### Default Engine Config
```yaml
tensor_parallel_size: 1
max_num_tokens: 8192
max_batch_size: 16
trust_remote_code: true
backend: pytorch
enable_chunked_prefill: true
kv_cache_config:
  free_gpu_memory_fraction: 0.85
cuda_graph_config:
  max_batch_size: 16
```

### High-Performance Engine Config
```yaml
tensor_parallel_size: 1
max_num_tokens: 16384
max_batch_size: 32
trust_remote_code: true
backend: pytorch
enable_chunked_prefill: true
kv_cache_config:
  free_gpu_memory_fraction: 0.90
cuda_graph_config:
  max_batch_size: 32
```

## NGC Authentication Setup

### Automatic Setup (via Deploy Script)
The deploy script will automatically create NGC secrets if you provide the API key:

```bash
# Set environment variable
export NGC_API_KEY=your-ngc-api-key-here
./deploy.sh trtllm-aggregated-default
```

### Manual Setup
```bash
# Get NGC API Key from https://ngc.nvidia.com
# Navigate to Setup → Generate API Key

# Create NGC secret
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password=your-ngc-api-key \
  -n dynamo-cloud
```

## Testing

### Basic Health Check
```bash
# Port forward to frontend service
kubectl port-forward service/trtllm-aggregated-default-frontend 8000:8000 -n dynamo-cloud

# Test health endpoint
curl http://localhost:8000/health
```

### Chat Completions
```bash
# Test inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "What is TensorRT-LLM?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Performance Testing
```bash
# Test with high-performance configuration
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Generate a detailed explanation of machine learning"}
    ],
    "max_tokens": 500,
    "temperature": 0.3
  }'
```

## Monitoring

### Pod Status
```bash
# Check deployment status
kubectl get dynamographdeployment trtllm-aggregated-default -n dynamo-cloud

# Check pods
kubectl get pods -n dynamo-cloud -l app=trtllm-aggregated-default
```

### Performance Monitoring
```bash
# GPU utilization (should be >80% during inference)
kubectl exec -it <trtllm-worker-pod> -n dynamo-cloud -- nvidia-smi

# Worker logs with performance metrics
kubectl logs -n dynamo-cloud -l componentType=worker -f | grep -i "throughput\|latency\|batch"
```

### Disaggregated Monitoring
```bash
# Check both prefill and decode workers
kubectl get pods -n dynamo-cloud -l app=trtllm-disaggregated-default

# Prefill worker logs
kubectl logs -n dynamo-cloud -l app=trtllm-disaggregated-default | grep prefill

# Decode worker logs
kubectl logs -n dynamo-cloud -l app=trtllm-disaggregated-default | grep decode
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
imagePullSecrets:
- name: ngc-secret  # Required for NGC authentication
```

### Recommended Instance Types
- **G5.2xlarge**: 1x A10G GPU (24GB) - optimal for Qwen3-0.6B
- **G5.4xlarge**: 1x A10G GPU (24GB) - for high-performance configs
- **G5.12xlarge**: 4x A10G GPU (96GB) - for multi-GPU tensor parallelism (future)

## Performance Tuning

### For Maximum Throughput
- Use `trtllm-aggregated-high-performance` configuration
- Monitor GPU utilization and adjust batch sizes
- Enable CUDA graphs for consistent performance
- Use 90% GPU memory utilization

### For Low Latency
- Use `trtllm-aggregated-default` with smaller batch sizes
- Enable chunked prefill for faster first token
- Optimize engine configuration for specific use case

### For Concurrent Requests
- Use `trtllm-disaggregated-default` architecture
- Scale prefill and decode workers independently
- Monitor cache transceiver performance

## External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments.

**TensorRT-LLM-Specific Notes:**
- Use Network Load Balancer (NLB) to minimize latency
- Consider `target-type: ip` for optimal TensorRT performance

## Cleanup

```bash
# Remove deployment
kubectl delete dynamographdeployment trtllm-aggregated-default -n dynamo-cloud
# or
kubectl delete dynamographdeployment trtllm-aggregated-high-performance -n dynamo-cloud
# or
kubectl delete dynamographdeployment trtllm-disaggregated-default -n dynamo-cloud
```

## Troubleshooting

### Common Issues

**NGC Authentication Failures:**
```bash
# Check NGC secret exists
kubectl get secret ngc-secret -n dynamo-cloud

# Check pod events for ImagePullBackOff
kubectl describe pod <trtllm-worker-pod> -n dynamo-cloud
```

**Model Loading Issues:**
```bash
# Check HuggingFace token
kubectl get secret hf-token-secret -n dynamo-cloud

# Monitor model download progress
kubectl logs <trtllm-worker-pod> -n dynamo-cloud -f | grep -i "download\|loading"
```

**Performance Issues:**
```bash
# Check GPU memory utilization
kubectl exec <trtllm-worker-pod> -n dynamo-cloud -- nvidia-smi

# Review engine configuration in logs
kubectl logs <trtllm-worker-pod> -n dynamo-cloud | grep -i "config\|batch\|memory"
```

**Disaggregated Communication Issues:**
```bash
# Check cache transceiver status
kubectl logs -n dynamo-cloud -l componentType=worker -f | grep -i "transceiver\|cache"

# Verify both workers are healthy
kubectl get pods -n dynamo-cloud -l app=trtllm-disaggregated-default -o wide
```

## References

- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [NGC Container Registry](https://catalog.ngc.nvidia.com/)
- [NVIDIA Dynamo Documentation](https://docs.nvidia.com/dynamo/)
- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
