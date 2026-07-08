---
sidebar_position: 3
---

# llm-d Cache-Aware Routing on EKS

:::warning
Deployment requires access to 8x `g5.2xlarge` GPU instances (NVIDIA A10G 24GB each). Costs approximately $10.49/hr. Scale nodes to zero when not testing.
:::

:::info
This blueprint uses [llm-d](https://llm-d.ai/) precise prefix-cache scheduling to reduce p90 TTFT by up to 96% compared to round-robin under sustained multi-turn load — with no model or application code changes.
:::

## What is cache-aware routing?

Standard Kubernetes load balancing scatters a user's successive inference requests across different vLLM pods, forcing each pod to recompute the full conversation context from scratch. Cache-aware routing maintains a real-time global index of KV-cache blocks across the fleet and routes each request to the pod with the highest prefix-cache affinity.

### Key Features and Benefits

- **Up to 96% reduction in p90 TTFT** under sustained 150-user multi-turn load
- **Precise prefix-cache scoring** via real-time ZeroMQ KV block introspection
- **No model or application code changes** — routing layer only
- **Gateway API native** — uses InferencePool CRD (v1 GA)
- **Security hardened** — TLS, non-root containers, NetworkPolicies, KMS-encrypted secrets
- **One-command deploy** via `setup.sh` (~25 minutes end-to-end)

### Components

| Component | Role | Version |
|-----------|------|---------|
| [vLLM](https://docs.vllm.ai/) | LLM inference engine with prefix caching + KVEvents | v0.22+ |
| [llm-d](https://llm-d.ai/) | Cache-aware request scheduler (CNCF Sandbox) | v0.5+ |
| [Envoy Gateway](https://gateway.envoyproxy.io/) | L7 proxy with ext-proc filter | v1.8.1 |
| [Envoy AI Gateway](https://aigateway.envoyproxy.io/) | InferencePool support controller | v1.0.0 |
| [cert-manager](https://cert-manager.io/) | TLS certificate provisioning | v1.17+ |
| [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) | InferencePool CRD | v1.5.0 |

## Architecture

![Architecture Overview](/img/inference/llm-d/architecture.png)

When a client sends an inference request:

1. The request arrives at the ALB and routes to Envoy Gateway
2. Envoy invokes the ext-proc gRPC filter, sending the request to the llm-d EPP
3. The EPP tokenizes the prompt, hashes it into 64-token block sequences, and queries the global KV-block index
4. The EPP scores all pods: prefix-cache affinity (weight 3) + queue depth (2) + KV-cache utilization (2) + LRU fallback (2)
5. The EPP returns the selected pod IP to Envoy, which forwards the request
6. The vLLM pod processes the request (skipping prefill for cached blocks) and streams the response
7. After processing, the pod publishes new KV blocks to ZeroMQ, updating the global index

## Prerequisites

- AWS account with quota for 8x `g5.2xlarge` (us-west-2)
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [kubectl](https://kubernetes.io/docs/tasks/tools/) v1.31+, [Helm](https://helm.sh/docs/intro/install/) v3.12+, [eksctl](https://eksctl.io/installation/)
- [Hugging Face token](https://huggingface.co/settings/tokens) with access to `mistralai/Mistral-7B-Instruct-v0.3`
- AWS KMS key for secrets encryption

## Deploying the Solution

### One-command deploy (~25 minutes)

```bash
git clone https://github.com/aws-samples/sample-eks-cache-aware-llm-routing.git
cd sample-eks-cache-aware-llm-routing

export HF_TOKEN=<your-huggingface-token>
export KMS_KEY_ARN=<your-kms-key-arn>
./scripts/setup.sh
```

The script:
1. Creates an EKS cluster with 8x g5.2xlarge GPU nodes (private networking, KMS encryption, audit logging)
2. Installs NVIDIA device plugin
3. Deploys 7 vLLM replicas with prefix caching + ZeroMQ KVEvents
4. Installs cert-manager, Gateway API CRDs, Envoy AI Gateway, and Envoy Gateway v1.8.1
5. Deploys llm-d router with precise prefix-cache scorer
6. Deploys benchmark runner and verifies both endpoints

### Step-by-step deployment

For a detailed walkthrough of each step, see the [repository README](https://github.com/aws-samples/sample-eks-cache-aware-llm-routing#deployment).

## Verify Deployment

```bash
# Check vLLM pods
kubectl -n inference get pods -l app=vllm-inference

# Check EPP
kubectl -n inference get pods -l llm-d-router-gateway=cache-aware-routing-epp

# Test both endpoints
CA_SVC=$(kubectl -n envoy-gateway-system get svc \
  -l gateway.envoyproxy.io/owning-gateway-name=inference-gateway \
  -o jsonpath='{.items[0].metadata.name}')

kubectl -n inference exec benchmark-runner -- python3 -c "
import urllib.request, json
endpoints = {
    'Round-Robin': 'http://vllm-inference.inference.svc.cluster.local:8000/v1/completions',
    'Cache-Aware': 'http://${CA_SVC}.envoy-gateway-system.svc.cluster.local:8080/v1/completions'
}
for name, ep in endpoints.items():
    req = urllib.request.Request(ep, data=json.dumps({'model':'mistralai/Mistral-7B-Instruct-v0.3','prompt':'Hello','max_tokens':5}).encode(), headers={'Content-Type':'application/json'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        print(f'{name}: OK')
    except Exception as e:
        print(f'{name}: FAIL - {e}')
"
```

## Benchmark Results

**Configuration**: 150 concurrent users, 25 QPS Poisson arrival, 7x vLLM pods (Mistral-7B on g5.2xlarge), 3-minute sustained multi-turn load.

| Time bucket | RR p90 | CA p90 | p90 Improvement |
|-------------|--------|--------|-----------------|
| 0–30s | 694ms | 209ms | **+70%** |
| 30–60s | 1,249ms | 274ms | **+78%** |
| 60–90s | 2,043ms | 195ms | **+90%** |
| 90–120s | 2,036ms | 152ms | **+93%** |
| 120–150s | 3,928ms | 147ms | **+96%** |
| 150–180s | 2,327ms | 146ms | **+94%** |

Round-robin p90 TTFT degrades to 3.9 seconds under sustained load while cache-aware routing holds at ~150ms.

### Run the benchmark yourself

```bash
kubectl cp benchmarks/sustained_benchmark.py inference/benchmark-runner:/tmp/bench.py
kubectl -n inference exec benchmark-runner -- python3 /tmp/bench.py
```

## When to use

| Workload | Cache-Aware | Standard LB |
|----------|:-----------:|:-----------:|
| Multi-turn conversations | ✅ | |
| Agentic workflows (growing context) | ✅ | |
| Multi-tenant (shared system prompts) | ✅ | |
| Stateless single-shot requests | | ✅ |
| Batch inference (latency not critical) | | ✅ |

## Cost

| Resource | Type | Qty | Cost/hr (us-west-2) |
|----------|------|-----|---------------------|
| EKS cluster | Control plane | 1 | $0.10 |
| GPU nodes | g5.2xlarge | 8 | $9.68 |
| NAT Gateway | Per-AZ | 1 | $0.045 |
| ALB | - | 1 | $0.022 |
| EBS volumes | gp3, 100GB | 8 | $0.64 |
| **Total** | | | **~$10.49/hr** |

## Clean Up

```bash
./scripts/cleanup.sh
# Or:
eksctl delete cluster --name cache-routing-benchmark --region us-west-2
```

## References

- [Sample code repository](https://github.com/aws-samples/sample-eks-cache-aware-llm-routing)
- [llm-d documentation](https://llm-d.ai/docs/getting-started)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
- [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- [Envoy AI Gateway](https://aigateway.envoyproxy.io/)
- [KV-Cache Wins You Can See (llm-d blog)](https://llm-d.ai/blog/kvcache-wins-you-can-see)
