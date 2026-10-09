---
title: "Qwen3 Coder 30B"
sidebar_label: "qwen3-coder-30b"
---

# Deploy Qwen3 Coder 30B on EKS

> Generated from `registry/models/qwen3-coder-30b.yaml`. Verified fields only.

## Overview

- **HF repo:** `Qwen/Qwen3-Coder-30B-A3B-Instruct`
- **Architecture:** `Qwen3MoeForCausalLM`
- **Parameters:** 30B (A3B MoE, ~3B active)
- **Precision:** bf16, fp8
- **Tensor parallel:** 4
- **Max model length:** 262144
- **Tool-call parser:** `qwen3_coder` (verified)
- **Status:** verified (2026-07-27)

## Verified configurations

| Instance       | Region tested | Tokens/sec | TTFT (ms) | ITL (ms) | Concurrency | $/1M tok |
| -------------- | ------------- | ---------: | --------: | -------: | ----------: | -------: |
| `g6e.12xlarge` | `us-east-2`   |      842.0 |       260 |     22.1 |           8 |     1.94 |

## Prerequisites

- An EKS cluster with Karpenter and the NVIDIA device plugin (see [`infra/`](https://github.com/awslabs/ai-on-eks/tree/main/infra)).
- A `g6e.12xlarge` NodePool (4x L40S).
- The `hf-token` Kubernetes secret:

  ```bash
  kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN
  ```

## Deploy

```bash
# 1. find the cheapest region with capacity
aoe-capacity find g6e.12xlarge --regions all

# 2. generate the blueprint from the registry entry
aoe-blueprint gen registry/models/qwen3-coder-30b.yaml --target vllm -o qwen3-coder-30b.yaml

# 3. apply and wait
kubectl apply -f qwen3-coder-30b.yaml
kubectl rollout status deployment/qwen3-coder-30b --timeout=20m

# 4. verify the endpoint
kubectl port-forward svc/qwen3-coder-30b 8000:8000 &
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Coder-30B-A3B-Instruct","messages":[{"role":"user","content":"Write a bubble sort in Python."}]}'
```

## Benchmark methodology

Numbers above come from an `inference-perf`-style concurrency sweep at 1/4/8 concurrent
requests against `http://localhost:8000/v1`, measuring tokens/sec, TTFT, and ITL. Cost
per 1M tokens = (g6e.12xlarge on-demand $/hr) / (tokens_per_sec \* 3600 / 1e6). Re-run the
benchmark in your own region and append rows to the registry entry.

## Cost

| Instance       | On-demand $/hr (us-east-2) | Tokens/sec @ conc 8 | $/1M tok |
| -------------- | -------------------------: | ------------------: | -------: |
| `g6e.12xlarge` |                     ~10.49 |               842.0 |     1.94 |

Spot pricing is typically 40-60% lower; run `aoe-capacity find g6e.12xlarge` for the
live number in each region.

## Troubleshooting

- **`CUDA out of memory`** — 262144 context at high concurrency can exhaust VRAM. Lower
  `--max-model-len` in the entry and regenerate, or move to a larger instance and raise
  `--tensor-parallel-size`.
- **Pod stuck `Pending`** — Karpenter has no `g6e.12xlarge` capacity in this region.
  Check `aoe-capacity find g6e.12xlarge --regions all` and deploy where it ranks highest.
- **401 pulling weights** — the `hf-token` secret is missing or lacks access; recreate it.
