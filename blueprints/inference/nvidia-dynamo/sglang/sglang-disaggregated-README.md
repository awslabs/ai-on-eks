# SGLang Disaggregated Example

Deploy SGLang with disaggregated serving architecture, combining SGLang's advanced caching with separate prefill/decode workers.

## Architecture

```text
Client Requests → Frontend → Decode Workers (RadixAttention)
                              ↑ NIXL Transfer ↓
                           Prefill Workers (Optimized)
```

This example demonstrates:
- **SGLang + Disaggregation**: Best of both worlds - advanced caching + specialized workers
- **RadixAttention**: SGLang's prefix tree caching with disaggregated serving
- **NIXL Integration**: Efficient KV transfers between SGLang workers
- **Advanced Memory Management**: SGLang's optimized memory pooling with disaggregation

## Why SGLang Disaggregated?

SGLang's RadixAttention provides excellent cache reuse, and when combined with disaggregated serving:

1. **Prefix Tree + Disaggregation**: Maximize cache benefits across specialized workers
2. **Memory Efficiency**: SGLang's memory pooling optimized for each phase
3. **Better Batching**: Dynamic batching strategies for prefill vs decode
4. **Cache Transfer**: Efficient movement of prefix tree data via NIXL

## Prerequisites

- Dynamo platform deployed in your EKS cluster
- `dynamo-cloud` namespace with secrets configured
- **Multi-GPU setup**: Minimum 2 GPUs for meaningful disaggregation
- HuggingFace token secret configured
- NIXL support for efficient KV cache transfer

## Deployment

Deploy using kubectl:

```bash
kubectl apply -f sglang-disagg.yaml -n dynamo-cloud
```

Or use the main deployment script:

```bash
cd ..
./deploy.sh sglang-disagg
```

## Model Configuration

This example uses `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` which works well with SGLang's caching strategies.

Alternative models:
- `Qwen/Qwen3-0.6B` - Smaller model for testing
- `meta-llama/Llama-3.1-8B-Instruct` - Good balance of performance and resource usage
- `mistralai/Mistral-7B-Instruct-v0.3` - Excellent cache reuse patterns

## Testing SGLang Disaggregation

### 1. Cache Reuse Test
Test SGLang's prefix tree caching:

```bash
# Port forward to frontend
# Port forward via Service (recommended) - enables both API access and metrics collection
kubectl port-forward service/sglang-disagg-frontend 8000:8000 -n dynamo-cloud

# Alternative: Direct deployment access
# kubectl port-forward deployment/sglang-disagg-frontend 8000:8000 -n dynamo-cloud

# Send requests with shared prefixes to trigger RadixAttention
SYSTEM_PROMPT="You are an expert in artificial intelligence and machine learning."

for topic in "neural networks" "deep learning" "transformers" "attention mechanisms" "backpropagation"; do
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
      "messages": [
        {"role": "system", "content": "'"$SYSTEM_PROMPT"'"},
        {"role": "user", "content": "Explain '"$topic"' in simple terms"}
      ],
      "max_tokens": 150
    }' > /tmp/response_'${topic// /_}'.json &
done
wait
```

### 2. Long Context Disaggregation
Test long prompts that trigger remote prefill:

```bash
# Generate long context that will be processed by prefill workers
LONG_CONTEXT=$(python3 -c "
import json
context = 'Context: ' + ' '.join([f'Important fact {i}: This is relevant information for the AI to consider.' for i in range(1, 100)])
print(json.dumps(context))
")

curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
      {"role": "user", "content": '$LONG_CONTEXT' + " Based on this context, what are the key points?"}
    ],
    "max_tokens": 200
  }'
```

### 3. Multi-turn with Prefix Tree
Test conversation continuity with RadixAttention:

```python
import requests
import json

base_url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Start with a detailed system prompt
conversation = [{
    "role": "system",
    "content": "You are an AI tutor specializing in computer science. Always provide detailed explanations with examples."
}]

# Multi-turn conversation to build up cache
topics = [
    "What is a hash table?",
    "How does collision resolution work in hash tables?",
    "What are the time complexities of hash table operations?",
    "Compare hash tables with binary search trees",
    "When should I use hash tables vs other data structures?"
]

for i, question in enumerate(topics):
    conversation.append({"role": "user", "content": question})

    response = requests.post(base_url, headers=headers, json={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "messages": conversation,
        "max_tokens": 150
    })

    assistant_reply = response.json()["choices"][0]["message"]["content"]
    conversation.append({"role": "assistant", "content": assistant_reply})

    print(f"Turn {i+1}: {len(conversation)} messages in conversation")
```

## Monitoring SGLang Disaggregation

### 1. Cache Performance
Monitor SGLang's cache efficiency:

```bash
# Check cache hit rates and prefix tree statistics
kubectl logs -n dynamo-cloud -l app=sglang-disagg -f | grep -E "(cache|hit|tree|radix)"

# Look for prefix tree growth
kubectl logs -n dynamo-cloud -l app=sglang-disagg-decode -f | grep -E "(prefix|tree|node)"

# Monitor memory efficiency
kubectl top pods -n dynamo-cloud -l app=sglang-disagg --containers
```

### 2. Disaggregation Metrics
Monitor disaggregated execution:

```bash
# Check prefill worker activity
kubectl logs -n dynamo-cloud -l app=sglang-disagg-prefill -f | grep -E "(prefill|NIXL|transfer)"

# Check decode worker routing decisions
kubectl logs -n dynamo-cloud -l app=sglang-disagg-decode -f | grep -E "(disagg|routing|local|remote)"

# Monitor overall system performance
kubectl logs -n dynamo-cloud -l app=sglang-disagg-frontend -f | grep -E "(TTFT|ITL|throughput)"
```

### 3. Worker Communication
Monitor NIXL transfers between workers:

```bash
# Check NIXL transfer logs
kubectl logs -n dynamo-cloud -l app=sglang-disagg -f | grep -i nixl

# Monitor prefill queue status
kubectl logs -n dynamo-cloud -l app=sglang-disagg -f | grep -E "(queue|pending)"
```

## Performance Benefits

### SGLang Advantages
1. **RadixAttention**: Automatic prefix sharing reduces redundant computation
2. **Advanced Batching**: Dynamic batching optimized for cache patterns
3. **Memory Pooling**: Efficient memory management reduces fragmentation

### Disaggregation Benefits
1. **Specialized Processing**: Prefill and decode optimized independently
2. **Better Resource Utilization**: Different parallelism for each phase
3. **Reduced Blocking**: Long prefills don't delay ongoing decode

### Combined Benefits
1. **Maximum Cache Efficiency**: Prefix tree + intelligent routing
2. **Optimal Resource Usage**: Best of caching + disaggregation
3. **Scalability**: Independent scaling with cache awareness

## Scaling SGLang Disaggregated

### Adaptive Scaling
```bash
# Scale prefill workers for high-variability workloads
kubectl patch dynamographdeployment sglang-disagg -n dynamo-cloud -p \
  '{"spec":{"services":{"SGLangPrefillWorker":{"replicas":3}}}}'

# Scale decode workers for high concurrent conversation scenarios
kubectl patch dynamographdeployment sglang-disagg -n dynamo-cloud -p \
  '{"spec":{"services":{"SGLangDecodeWorker":{"replicas":4}}}}'
```

### Cache Optimization
Monitor cache effectiveness and adjust based on workload:

```bash
# Check cache hit rates to determine optimal worker distribution
kubectl logs -n dynamo-cloud -l app=sglang-disagg -f | grep -E "hit.rate|cache.efficiency"
```

## Troubleshooting

### Cache-Related Issues
1. **Low Cache Hit Rate**: Check if prefix patterns are being recognized
2. **Memory Pressure**: Monitor for prefix tree eviction events
3. **Transfer Inefficiency**: Verify NIXL is working for cache transfers

### Disaggregation Issues
1. **Queue Backlog**: Scale up prefill workers if queue grows
2. **NIXL Failures**: Check GPU connectivity and memory allocation
3. **Worker Discovery**: Verify ETCD registration and NATS communication

## External Access

For production external access, see the main README.md **External Access** section which provides comprehensive guidance for all Dynamo deployments.

**Note**: This applies to all Dynamo deployments including disaggregated architectures.


## Cleanup

Remove the deployment:

```bash
kubectl delete dynamographdeployment sglang-disagg -n dynamo-cloud
```
