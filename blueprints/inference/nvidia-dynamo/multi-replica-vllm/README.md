# Multi-Replica vLLM Example

Deploy high-availability vLLM with multiple worker replicas, disaggregated serving, and KV-aware routing using Dynamo v0.5.0.

> **Important**: This example provides **high availability and load balancing** using multiple independent worker replicas (each running the full model). For true multi-node tensor parallelism (splitting large models across nodes), see the [Multi-node Limitations](#multi-node-limitations) section below.

## Architecture

```text
Client Requests → Frontend (KV Router) → Prefill Workers → Decode Workers
                                              ↓ NIXL Transfer ↓
                                           Disaggregated KV Cache
```

This example demonstrates:
- **Multiple Worker Replicas**: Independent workers for high availability and load distribution
- **Disaggregated Serving**: Separate prefill and decode workers per replica
- **KV-aware Routing**: Intelligent request routing based on cache overlap
- **NIXL GPU-to-GPU Transfers**: Efficient KV cache transfers between prefill and decode workers

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- **Multi-GPU setup**: Minimum 2 GPUs for disaggregated serving
- HuggingFace token secret configured
- NIXL support for efficient KV cache transfer

## Deployment

Deploy using kubectl:

```bash
kubectl apply -f multi-replica-vllm.yaml -n dynamo-cloud
```

Or use the main deployment script:

```bash
cd ..
./deploy.sh multi-replica-vllm
```

## Model Configuration

This example uses `Qwen/Qwen3-0.6B` for demonstration. For production workloads, consider larger models that benefit more from disaggregation:

- `meta-llama/Llama-3.1-8B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

**Note**: Each worker replica runs the **complete model** independently. For very large models (70B+), consider single-node deployment or wait for true multi-node tensor parallelism support.

## Key Features

### Disaggregated Serving
- **Prefill Workers**: Optimized for parallel processing of input tokens
- **Decode Workers**: Optimized for sequential token generation
- **Independent Scaling**: Scale prefill and decode workers based on workload

### KV-Aware Routing
- Routes requests to workers with best cache overlap
- Maximizes cache reuse for improved performance
- Automatic load balancing with cache awareness
- Tracks cached data across ALL workers (both prefill and decode)

### Conditional Disaggregation
- **Smart Routing**: Decode workers decide at runtime whether to handle prefill locally or remotely
- **Short Prefills**: Handled locally for better latency
- **Long Prefills**: Sent to dedicated prefill workers to avoid blocking decode operations
- **Automatic Fallback**: System continues operating even if prefill workers are unavailable

## Testing

Once deployed, test the multi-replica vLLM service:

```bash
# Port forward to the frontend service (with KV routing)
# Port forward via Service (recommended) - enables both API access and metrics collection
kubectl port-forward service/multi-replica-vllm-frontend 8000:8000 -n dynamo-cloud

# Alternative: Direct deployment access
# kubectl port-forward deployment/multi-replica-vllm-frontend 8000:8000 -n dynamo-cloud

# Test health endpoint
curl http://localhost:8000/health

# Test chat completions (will be routed optimally)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain disaggregated serving benefits"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'

# Test multi-turn conversation (benefits from KV routing)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
      {"role": "user", "content": "What is my name?"}
    ],
    "max_tokens": 50
  }'
```

## Monitoring

Monitor the multi-replica deployment:

```bash
# View all pods
kubectl get pods -n dynamo-cloud -l app=multi-replica-vllm

# Check prefill worker logs
kubectl logs -n dynamo-cloud -l app=multi-replica-vllm-prefill -f

# Check decode worker logs
kubectl logs -n dynamo-cloud -l app=multi-replica-vllm-decode -f

# Check frontend logs for routing decisions (with DYN_LOG=debug)
kubectl logs -n dynamo-cloud -l app=multi-replica-vllm-frontend -f | grep -i "routing\|overlap"

# Monitor NIXL transfers
kubectl logs -n dynamo-cloud -l app=multi-replica-vllm -f | grep -i "nixl\|transfer"
```

## Performance Benefits

Multi-replica disaggregated serving provides:

1. **Better Resource Utilization**: Separate optimization for compute vs memory-bound phases
2. **Improved Latency**: No head-of-line blocking between prefill and decode
3. **Enhanced Scalability**: Independent scaling of prefill and decode capacity
4. **KV Cache Efficiency**: Intelligent routing maximizes cache reuse

## Scaling

Scale workers independently based on workload:

```bash
# Scale prefill workers for high input throughput
kubectl patch dynamographdeployment multi-replica-vllm -n dynamo-cloud -p '{"spec":{"services":{"PrefillWorker":{"replicas":3}}}}'

# Scale decode workers for more concurrent generations
kubectl patch dynamographdeployment multi-replica-vllm -n dynamo-cloud -p '{"spec":{"services":{"DecodeWorker":{"replicas":4}}}}'
```

## Troubleshooting

Common issues and solutions:

1. **NIXL Transfer Failures**: Check GPU connectivity between nodes
2. **KV Routing Not Working**: Verify frontend is using `--router-mode kv`
3. **Workers Not Discovering**: Check etcd connectivity and service registration
4. **Performance Issues**: Monitor GPU utilization and adjust worker ratios

## Multi-node Limitations

### What This Example Provides
- **Multiple Independent Workers**: Each worker replica runs the complete model (TP=1)
- **Load Balancing**: Requests distributed across multiple workers for throughput
- **High Availability**: Service continues if individual workers fail
- **KV-aware Routing**: Intelligent request distribution based on cache overlap

### What This Example Does NOT Provide
- **Tensor Parallelism Across Nodes**: Models are not split across nodes
- **Memory Scaling**: Large models (70B+) require each worker to fit the full model
- **Cross-node Model Sharding**: Each replica loads the complete model independently

### True Multi-node Tensor Parallelism
For actual multi-node tensor parallelism (splitting models across nodes), Dynamo requires:

- **Slurm/MPI Environment**: Uses `mpirun` or `srun` for distributed model loading
- **Examples Available**: TRT-LLM with WideEP, SGLang with `--nnodes` flags
- **Kubernetes Support**: Not currently supported in Kubernetes deployments
- **Future Development**: Cross-node tensor parallelism in K8s may be added in future versions

**For large models requiring cross-node parallelism, consider:**
1. Using Slurm-based deployments (see Dynamo upstream docs)
2. Single-node deployments with high-memory instances
3. Waiting for future K8s multi-node tensor parallelism support

## External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments.

**Note**: This applies to all Dynamo deployments including disaggregated architectures.


## Cleanup

Remove the deployment:

```bash
kubectl delete dynamographdeployment multi-replica-vllm -n dynamo-cloud
```
