---
sidebar_position: 3
sidebar_label: llm-d Cache-Aware Routing
---

# llm-d Cache-Aware Routing on EKS

:::warning
Deployment requires 8x `g5.2xlarge` GPU instances (NVIDIA A10G 24GB each). Costs approximately **$10.49/hr**. Scale nodes to zero when not testing.
:::

This solution deploys **precise KV-cache-aware routing** using [llm-d](https://llm-d.ai/) (CNCF Sandbox) on Amazon EKS, achieving **up to 96% reduction in p90 TTFT** under sustained multi-turn load compared to round-robin, with no model or application code changes.

## Why?

When you scale vLLM from one pod to many, prefix caching stops working. Not because it's broken. It still works perfectly on each individual pod. The problem is that round-robin routing sends Turn 5 of a conversation to a pod that never saw Turns 1 through 4. That pod has to recompute the entire conversation prefix from scratch, even though another pod already has it cached in GPU memory.

This gets worse over time. The longer a conversation runs, the bigger the prefix that gets recomputed on each cache miss. In our benchmarks with 150 concurrent users, round-robin p90 TTFT degrades from under 700ms to nearly 4 seconds within 3 minutes. Purely from wasted prefill, not from GPU saturation.

## Use Cases

- **Multi-turn chatbots**: Response times stay flat regardless of conversation length
- **Agentic workflows**: Agent loop iterations hit cache on the pod holding prior context (100:1 input-to-output ratios)
- **Multi-tenant SaaS**: Shared system prompts (6,000+ tokens) cached once, reused across users on the same pod
- **RAG pipelines**: Repeated context documents served from cache instead of recomputed

For stateless single-shot requests without prefix reuse, standard load balancing remains appropriate.

## How llm-d approaches this

llm-d offers three deployment paths, each solving a different inference bottleneck:

- **Intelligent Inference Scheduling**: Predicts which pod likely has the prefix cached based on routing history. No vLLM changes needed. Good for getting started, but predictions degrade under dynamic load.

- **Precise Prefix-Cache Scheduling** (this solution): vLLM publishes real-time KV block events over ZeroMQ. The router knows *exactly* which blocks are on which pod. Maximum cache hit rate, validated at 96% p90 improvement.

- **Prefill/Decode Disaggregation**: Separates the expensive prefill phase onto dedicated pods, with decode running separately. For very long contexts (32K+) and batch-heavy workloads.

## Architecture

The solution combines [vLLM](https://docs.vllm.ai/) for inference, [llm-d](https://llm-d.ai/) for routing decisions, [Envoy Gateway](https://gateway.envoyproxy.io/) as the L7 proxy, [Envoy AI Gateway](https://aigateway.envoyproxy.io/) for InferencePool support, and [cert-manager](https://cert-manager.io/) for TLS.

![Architecture Diagram](/img/inference/llm-d/architecture.png)

The architecture operates across four layers:

1. **Ingress layer**: An Application Load Balancer receives client requests and forwards to Envoy Gateway. Envoy invokes the ext-proc gRPC filter for routing decisions.

2. **Routing intelligence layer**: The llm-d Endpoint Picker (EPP) tokenizes the prompt, queries the global KV-block index, scores candidate pods, and returns a routing decision.

3. **Inference layer**: Seven vLLM pods serve the model with prefix caching active. Each pod maintains its own GPU KV-cache and exposes an OpenAI-compatible API.

4. **Feedback loop**: Each vLLM pod publishes KV block create/evict events over ZeroMQ. The EPP subscribes and continuously updates the global index.

Envoy AI Gateway and cert-manager are control-plane components (not on the data path).

### Request flow

1. Request arrives at ALB → routes to Envoy Gateway (port 8080)
2. Envoy invokes [ext-proc](https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/ext_proc_filter) gRPC → sends request to llm-d EPP
3. EPP tokenizes prompt, hashes into 64-token block sequences
4. EPP scores all pods: prefix-cache affinity (weight 3) + queue depth (2) + KV-cache utilization (2) + LRU fallback (2)
5. EPP returns selected pod IP → Envoy forwards request
6. vLLM pod processes request (skipping prefill for cached blocks) → streams response
7. Pod publishes new KV blocks to [ZeroMQ](https://zeromq.org/) → updates global index

The following diagrams illustrate the cache-hit and cache-miss scenarios:

![Cache Hit Flow](/img/inference/llm-d/cache-hit-flow.png)

*Cache hit: Turn 3 routes to Pod 3 (94% prefix match). Prefill skipped for cached blocks..*

![Cache Miss Flow](/img/inference/llm-d/cache-miss-flow.png)

*Cache miss: New user, no cached blocks. EPP falls back to load-aware scoring. Full prefill required.. Future requests from this user cache-hit on the assigned pod.*

### How precise scoring works

Each vLLM pod publishes KV block events over ZeroMQ (port 5556). The EPP subscribes to all pods, building a global block-hash-to-pod index. On each request, the EPP determines what percentage of the prompt's prefix already resides on each pod and scores accordingly.

Index overhead: ~339 KB to track a full cluster (1,000,000:1 data-to-metadata ratio).

### Components

| Component | Role | Version |
|-----------|------|---------|
| [vLLM](https://docs.vllm.ai/) | Inference engine with prefix caching + KVEvents | v0.22+ |
| [llm-d](https://llm-d.ai/) | Cache-aware request scheduler (CNCF Sandbox) | v0.5+ |
| [Envoy Gateway](https://gateway.envoyproxy.io/) | L7 proxy with ext-proc filter | v1.8.1 |
| [Envoy AI Gateway](https://aigateway.envoyproxy.io/) | InferencePool support controller | v1.0.0 |
| [cert-manager](https://cert-manager.io/) | TLS certificates | v1.17+ |
| [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/) | InferencePool CRD | v1.5.0 |

## Deployment

### Prerequisites

- AWS account with quota for 8x `g5.2xlarge` (us-west-2)
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), [kubectl](https://kubernetes.io/docs/tasks/tools/) v1.31+, [Helm](https://helm.sh/docs/intro/install/) v3.12+, [eksctl](https://eksctl.io/installation/)
- [Hugging Face token](https://huggingface.co/settings/tokens) with access to `mistralai/Mistral-7B-Instruct-v0.3`
- AWS KMS key for secrets encryption

### Deploy (~25 minutes)

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/llm-d-cache-aware-routing

export HF_TOKEN=<your-huggingface-token>
export KMS_KEY_ARN=<your-kms-key-arn>
./scripts/setup.sh
```

The script performs:

1. Creates EKS v1.31 cluster with 8x g5.2xlarge GPU nodes (private networking, KMS encryption, audit logging)
2. Installs NVIDIA device plugin
3. Creates namespace and secrets
4. Deploys 7 vLLM replicas with prefix caching + ZeroMQ KVEvents
5. Installs cert-manager, Gateway API CRDs, Envoy AI Gateway, Envoy Gateway v1.8.1
6. Deploys llm-d router with precise prefix-cache scorer
7. Deploys benchmark runner and verifies both endpoints

### Validate

```bash
# Check vLLM pods (7 Running)
kubectl -n inference get pods -l app=vllm-inference

# Check EPP
kubectl -n inference get pods -l llm-d-router-gateway=cache-aware-routing-epp

# Check Gateway
kubectl -n envoy-gateway-system get gateway inference-gateway

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

## Benchmark Results

**Configuration**: 150 concurrent users, 25 QPS Poisson arrival, 7x vLLM pods (Mistral-7B on g5.2xlarge), 3-minute sustained multi-turn load.

| Time Bucket | Round-Robin p90 | Cache-Aware p90 | Improvement |
|-------------|-----------------|-----------------|-------------|
| 0–30s | 694ms | 209ms | **+70%** |
| 30–60s | 1,249ms | 274ms | **+78%** |
| 60–90s | 2,043ms | 195ms | **+90%** |
| 90–120s | 2,036ms | 152ms | **+93%** |
| 120–150s | 3,928ms | 147ms | **+96%** |
| 150–180s | 2,327ms | 146ms | **+94%** |

### Why the improvement grows over time

With 7 pods and round-robin, each turn has an 85.7% probability of landing on a pod without prior context. As conversations grow, the wasted prefill per miss increases. Cache-aware routing maintains pod affinity. Turn 5 only prefills the small delta since Turn 4, regardless of total conversation length.

### Run the benchmark

```bash
kubectl cp benchmarks/sustained_benchmark.py inference/benchmark-runner:/tmp/bench.py
kubectl -n inference exec benchmark-runner -- python3 /tmp/bench.py
```

Takes ~12 minutes (5 min per strategy + cooldown).

## Security

- **Private networking**: GPU nodes in private subnets, no public IPs
- **API server**: Public + private endpoints (nodes communicate privately)
- **Secrets encryption**: KMS via `secretsEncryption.keyARN`
- **Audit logging**: API, audit, authenticator, controller-manager, scheduler → CloudWatch
- **TLS on EPP**: ext-proc gRPC with `--secure-serving=true`
- **Non-root containers**: UID 1000, `allowPrivilegeEscalation: false`, all capabilities dropped
- **Resource limits**: Set on all containers
- **NetworkPolicy**: ZMQ port 5556 restricted to EPP pods only
- **HF token**: Injected as env var from Secret, never in process args

## Monitoring

| Metric | Source | Normal Range |
|--------|--------|--------------|
| `vllm_num_requests_waiting` | vLLM | < 50 |
| `vllm_gpu_cache_usage_perc` | vLLM | 60-90% |
| `vllm_e2e_request_latency_seconds` | vLLM | p95 < 30s |
| `epp_cache_hit_ratio` | EPP | > 70% (multi-turn) |
| `epp_routing_decision_latency_ms` | EPP | < 5ms |

Scale on `vllm_num_requests_waiting` and `vllm_gpu_cache_usage_perc`, not CPU/memory.

For HA, run 2+ EPP replicas (active-active, independent ZMQ subscriptions). InferencePool `failureMode: FailOpen` falls back to round-robin if EPP is unavailable.

## Migration Path

| Phase | Configuration | Suitable for |
|-------|--------------|--------------|
| Phase 1 | llm-d (approximate scoring) | PoC, dev, quick validation |
| **Phase 2** | **llm-d (precise scoring)** | **Production with TTFT SLOs (this solution)** |
| Phase 3 | llm-d (precise + disaggregated P/D) | Mid-scale, prefill/decode separation |
| Phase 4 | NVIDIA Dynamo | Large scale (64+ GPUs) |

Phase 1 → 2: Add `--kv-events-config` to vLLM, redeploy EPP with precise scorer. No cluster changes needed.

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

Scale to zero without deleting:

```bash
aws eks update-nodegroup-config --cluster-name cache-routing-benchmark \
  --nodegroup-name gpu-nodes --scaling-config minSize=0,maxSize=8,desiredSize=0 \
  --region us-west-2
```

## Troubleshooting

For detailed troubleshooting (pods pending, EPP not connecting, gateway errors, model loading issues), see the [solution README](https://github.com/awslabs/ai-on-eks/tree/main/infra/solutions/llm-d-cache-aware-routing#troubleshooting).

## References

- [llm-d Documentation](https://llm-d.ai/docs/getting-started)
- [Gateway API Inference Extension](https://gateway-api-inference-extension.sigs.k8s.io/)
- [Envoy AI Gateway](https://aigateway.envoyproxy.io/)
- [vLLM Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
- [KV-Cache Wins You Can See (llm-d blog)](https://llm-d.ai/blog/kvcache-wins-you-can-see)
- [Amazon EKS User Guide](https://docs.aws.amazon.com/eks/latest/userguide/)
