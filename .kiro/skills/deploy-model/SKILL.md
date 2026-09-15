---
name: deploy-model
description: Deploy a registered open-weight model on EKS. Use when the user asks to deploy a model (e.g. "deploy qwen3-coder-30b"). Resolves the registry entry, checks capacity, generates a blueprint, applies it, and verifies the endpoint.
---

# Deploy Model on EKS

When the user asks to deploy an open-weight model on EKS:

1. Resolve `registry/models/<name>.yaml`. If absent, do the best-effort sizing
   walk-through (params -> VRAM -> instance count) and flag results unverified; never
   invent benchmark numbers.
2. Read the verified `instances[].type`, then `aoe-capacity find <instance_type> --regions all`
   and pick the cheapest ranked region.
3. Ensure the `hf-token` secret exists (`kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN`).
4. `aoe-blueprint gen registry/models/<name>.yaml --target vllm -o /tmp/<name>.yaml`.
5. `kubectl apply -f /tmp/<name>.yaml`.
6. `kubectl rollout status deployment/<name> --timeout=20m`, port-forward, and curl
   `/v1/chat/completions`. A JSON completion confirms success; otherwise use the
   troubleshoot-inference skill.
