# AGENTS.md — ai-on-eks

Entry point for generic coding agents (Codex and others). This repo ships four agent
capabilities for deploying open-weight models on EKS. The full instructions live in
`.claude/skills/<skill>/SKILL.md` (Claude Code) and `.kiro/skills/<skill>/SKILL.md` (Kiro);
this file is the provider-neutral summary.

## Capabilities

- **deploy-model** — "deploy qwen3-coder-30b": read `registry/models/<name>.yaml`,
  `aoe-capacity find <instance_type>`, ensure the `hf-token` secret,
  `aoe-blueprint gen ... | kubectl apply -f -`, then verify with a
  `/v1/chat/completions` curl. Unregistered models get an unverified sizing estimate.
- **find-capacity** — "where can I get 8xH100 cheapest?": run
  `aoe-capacity find <instance_type> --regions all` and report the ranked list.
- **benchmark-model** — concurrency sweep against a deployed endpoint; report
  tokens_per_sec, ttft_ms, itl_ms, cost_per_1m_tok_usd.
- **troubleshoot-inference** — diagnose OOM, CUDA mismatch, Karpenter scaling, and
  gated-repo/HF-token failures.

## Tools

- `aoe-capacity` — `tools/capacity/` (public AWS APIs only).
- `aoe-blueprint` — `tools/blueprintgen/` (registry entry -> Kubernetes manifest).
- `python -m registry.validate` — validate registry entries.

## Rules

- Facts are verified-or-unset. Never invent benchmark numbers or a `tool_parser`.
- Additive only: never restructure existing blueprints or Terraform.
