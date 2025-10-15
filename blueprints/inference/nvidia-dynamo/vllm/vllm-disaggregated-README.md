# vLLM Disaggregated Example

Deploy vLLM with disaggregated serving architecture - separate prefill and decode workers with NIXL GPU-to-GPU transfers.

## Architecture

```text
Client Requests → Frontend → Decode Workers
                              ↑ NIXL Transfer ↓
                           Prefill Workers
```

This example demonstrates:
- **Disaggregated Serving**: Separate prefill (compute-bound) and decode (memory-bound) phases
- **NIXL Transfers**: Efficient GPU-to-GPU KV cache transfers
- **Conditional Disaggregation**: Smart routing between local and remote prefill
- **Independent Scaling**: Scale prefill and decode workers independently

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- **Multi-GPU setup**: Minimum 2 GPUs (1 for prefill, 1 for decode)
- HuggingFace token secret configured
- NIXL support for GPU-to-GPU communication

## Key Benefits

### Performance Optimizations
1. **No Head-of-Line Blocking**: Long prefills don't block ongoing decode operations
2. **Specialized Workers**: Each phase optimized for its computational characteristics
3. **Better GPU Utilization**: Different parallelism strategies for prefill vs decode
4. **Efficient Transfers**: Direct GPU-to-GPU KV cache transfers via NIXL

### Smart Disaggregation
- **Conditional Routing**: Short prefills processed locally for efficiency
- **Queue Management**: Load balancing across multiple prefill workers
- **Runtime Reconfigurable**: Add/remove workers without system downtime

## Deployment

Deploy using kubectl:

```bash
kubectl apply -f vllm-disagg.yaml -n dynamo-cloud
```

Or use the main deployment script:

```bash
cd ..
./deploy.sh vllm-disagg
```

## Model Configuration

This example uses `Qwen/Qwen3-0.6B` for demonstration. For production workloads that benefit more from disaggregation:

- `meta-llama/Llama-3.1-8B-Instruct` - Better showcases prefill/decode separation
- `meta-llama/Llama-3.1-70B-Instruct` - Requires multiple GPUs, ideal for disaggregation
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` - Good performance characteristics

## Testing

Test the disaggregated serving:

```bash
# Port forward to frontend
# Port forward via Service (recommended) - enables both API access and metrics collection
kubectl port-forward service/vllm-disagg-frontend 8000:8000 -n dynamo-cloud

# Alternative: Direct deployment access
# kubectl port-forward deployment/vllm-disagg-frontend 8000:8000 -n dynamo-cloud

# Test health endpoint
curl http://localhost:8000/health

# Test short prompt (likely processed locally)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'

# Test long context (triggers disaggregation)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "'"$(python3 -c "print('Please analyze this long text: ' + 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' * 50)")"'"}],
    "max_tokens": 100
  }'
```

## Monitoring Disaggregation

Monitor the disaggregated architecture:

```bash
# Check all pods
kubectl get pods -n dynamo-cloud -l app=vllm-disagg

# Monitor prefill worker logs (look for NIXL transfers)
kubectl logs -n dynamo-cloud -l app=vllm-disagg-prefill -f | grep -E "(NIXL|transfer|remote)"

# Monitor decode worker logs (look for routing decisions)
kubectl logs -n dynamo-cloud -l app=vllm-disagg-decode -f | grep -E "(disagg|remote|local)"

# Check system resource usage
kubectl top pods -n dynamo-cloud -l app=vllm-disagg --containers
```

## Performance Tuning

### Disaggregation Thresholds
The system uses smart thresholds to decide between local and remote prefill:

1. **Prefill Length Threshold**: Short prefills processed locally
2. **Queue Size Threshold**: Avoids overloading prefill workers
3. **Prefix Cache Considerations**: High cache hit rates favor local processing

### Scaling Strategies
```bash
# Scale prefill workers for high input throughput workloads
kubectl patch dynamographdeployment vllm-disagg -n dynamo-cloud -p \
  '{"spec":{"services":{"VllmPrefillWorker":{"replicas":3}}}}'

# Scale decode workers for high concurrent request scenarios
kubectl patch dynamographdeployment vllm-disagg -n dynamo-cloud -p \
  '{"spec":{"services":{"VllmDecodeWorker":{"replicas":4}}}}'
```

## Troubleshooting

### Common Issues

1. **NIXL Transfer Failures**
   - Check GPU connectivity between nodes
   - Verify CUDA IPC is properly configured
   - Ensure sufficient GPU memory for KV cache transfers

2. **High Latency**
   - Monitor prefill queue size - scale up prefill workers if needed
   - Check for network bottlenecks between prefill and decode workers
   - Verify conditional disaggregation is working (check logs)

3. **Workers Not Communicating**
   - Verify ETCD connectivity for metadata sharing
   - Check NATS prefill queue functionality
   - Ensure proper service discovery between components

### Debug Commands
```bash
# Check NIXL metadata in ETCD
kubectl exec -n dynamo-cloud deployment/etcd -- etcdctl get --prefix /dynamo/nixl

# Monitor prefill queue
kubectl logs -n dynamo-cloud -l app=vllm-disagg-prefill -f | grep -i queue

# Check disaggregation routing decisions
kubectl logs -n dynamo-cloud -l app=vllm-disagg-decode -f | grep -i "routing\|disagg"
```

## External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments.

**Note**: This applies to all Dynamo deployments including disaggregated architectures.


## Cleanup

Remove the deployment:

```bash
kubectl delete dynamographdeployment vllm-disagg -n dynamo-cloud
```
