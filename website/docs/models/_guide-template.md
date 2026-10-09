---
title: "<Model Display Name>"
sidebar_label: "<model-name>"
draft: true
---

# Deploy <Model Display Name> on EKS

> Generated from `registry/models/<name>.yaml`. Verified fields only — anything not
> benchmarked is omitted, never guessed.

## Overview

- **HF repo:** `<hf_repo>`
- **Architecture:** `<architecture>`
- **Parameters:** `<params_b>`B
- **Precision:** `<precision>`
- **Tensor parallel:** `<tensor_parallel>`
- **Max model length:** `<max_model_len>`
- **Status:** `<status>` (verified <verified_date>)

## Verified configurations

| Instance | Region tested | Tokens/sec | TTFT (ms) | ITL (ms) | Concurrency | $/1M tok |
| -------- | ------------- | ---------: | --------: | -------: | ----------: | -------: |
| `<type>` | `<region>`    |    `<tps>` |  `<ttft>` |  `<itl>` |    `<conc>` | `<cost>` |

## Prerequisites

- An EKS cluster with Karpenter and the NVIDIA device plugin (see `infra/`).
- The `hf-token` Kubernetes secret:
  `kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN`.

## Deploy

```bash
aoe-capacity find <instance_type> --regions all      # pick cheapest region
aoe-blueprint gen registry/models/<name>.yaml --target vllm -o <name>.yaml
kubectl apply -f <name>.yaml
kubectl rollout status deployment/<name> --timeout=20m
```

## Benchmark methodology

Concurrency sweep at 1/4/8 against `http://localhost:8000/v1` capturing tokens/sec,
TTFT, and ITL; cost/1M tok = (instance $/hr) / (tokens_per_sec \* 3600 / 1e6).

## Troubleshooting

See the troubleshoot-inference skill: OOM -> raise TP or lower max-model-len; Pending ->
Karpenter/capacity; 401 -> `hf-token` secret.
