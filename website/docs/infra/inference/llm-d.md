---
sidebar_position: 3
sidebar_label: llm-d Cache-Aware Routing
---

# llm-d Cache-Aware Routing on EKS

:::warning
Deployment requires access to 8x `g5.2xlarge` GPU instances (NVIDIA A10G 24GB each). Costs approximately **$10.49/hr**. Scale nodes to zero when not testing.
:::

:::info
This solution achieves **up to 96% reduction in p90 TTFT** compared to round-robin under sustained multi-turn load — with no model or application code changes.
:::

### What is cache-aware routing?

Standard Kubernetes load balancing scatters requests across vLLM pods without awareness of GPU KV-cache state. Cache-aware routing maintains a real-time global index of KV-cache blocks and routes each request to the pod with the highest prefix-cache affinity, eliminating redundant prefill computation.

### Key Features and Benefits

- Precise prefix-cache scoring via real-time ZeroMQ KV block introspection
- Gateway API native (InferencePool CRD v1 GA)
- No model or application code changes required
- Security hardened (TLS, non-root, NetworkPolicies, KMS encryption)
- One-command deploy (~25 minutes end-to-end)

### Components

| Component | Role | Version |
|-----------|------|---------|
| [vLLM](https://docs.vllm.ai/) | Inference engine with prefix caching + KVEvents | v0.22+ |
| [llm-d](https://llm-d.ai/) | Cache-aware request scheduler (CNCF Sandbox) | v0.5+ |
| [Envoy Gateway](https://gateway.envoyproxy.io/) | L7 proxy with ext-proc filter | v1.8.1 |
| [Envoy AI Gateway](https://aigateway.envoyproxy.io/) | InferencePool support controller | v1.0.0 |
| [cert-manager](https://cert-manager.io/) | TLS certificates | v1.17+ |
| [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) | InferencePool CRD | v1.5.0 |

## Deploying the Solution

import CollapsibleContent from '@site/src/components/CollapsibleContent';

<CollapsibleContent header={<h3><span>Deploy the Solution</span></h3>}>

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/llm-d-cache-aware-routing

export HF_TOKEN=<your-huggingface-token>
export KMS_KEY_ARN=<your-kms-key-arn>
./scripts/setup.sh
```

For detailed step-by-step instructions, architecture explanation, troubleshooting, and benchmark methodology, see the [full README](https://github.com/awslabs/ai-on-eks/tree/main/infra/solutions/llm-d-cache-aware-routing).

</CollapsibleContent>

<CollapsibleContent header={<h3><span>Validate the Deployment</span></h3>}>

```bash
# Check pods
kubectl -n inference get pods -l app=vllm-inference
kubectl -n inference get pods -l llm-d-router-gateway=cache-aware-routing-epp

# Test both endpoints
CA_SVC=$(kubectl -n envoy-gateway-system get svc \
  -l gateway.envoyproxy.io/owning-gateway-name=inference-gateway \
  -o jsonpath='{.items[0].metadata.name}')

kubectl -n inference exec benchmark-runner -- python3 -c "
import urllib.request, json
for name, ep in [('Round-Robin','http://vllm-inference.inference.svc.cluster.local:8000/v1/completions'),
                 ('Cache-Aware','http://${CA_SVC}.envoy-gateway-system.svc.cluster.local:8080/v1/completions')]:
    req = urllib.request.Request(ep, data=json.dumps({'model':'mistralai/Mistral-7B-Instruct-v0.3','prompt':'Hello','max_tokens':5}).encode(), headers={'Content-Type':'application/json'})
    try: print(f'{name}: {urllib.request.urlopen(req, timeout=30).status}')
    except Exception as e: print(f'{name}: FAIL - {e}')
"
```

</CollapsibleContent>

<CollapsibleContent header={<h3><span>Clean Up</span></h3>}>

```bash
./scripts/cleanup.sh
# Or:
eksctl delete cluster --name cache-routing-benchmark --region us-west-2
```

</CollapsibleContent>

### Benchmark Results

150 concurrent users, 25 QPS, 7x vLLM pods (Mistral-7B on g5.2xlarge), 3-minute sustained multi-turn load:

| Time Bucket | Round-Robin p90 | Cache-Aware p90 | Improvement |
|-------------|-----------------|-----------------|-------------|
| 0–30s | 694ms | 209ms | **+70%** |
| 60–90s | 2,043ms | 195ms | **+90%** |
| 120–150s | 3,928ms | 147ms | **+96%** |

Round-robin degrades to 3.9s while cache-aware routing holds at ~150ms.
