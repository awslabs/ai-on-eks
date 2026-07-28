---
name: benchmark-model
description: Benchmark a deployed inference endpoint (throughput, TTFT, ITL, cost). Use when the user wants to measure a running model endpoint. Runs a concurrency sweep against an OpenAI-compatible endpoint on the cluster.
---

# Benchmark a Deployed Model

When the user wants to benchmark a running endpoint:

1. Confirm the deployment is up; port-forward `svc/<name>` to localhost:8000.
2. Run a 2-3 point concurrency sweep against `http://localhost:8000/v1`, capturing
   tokens_per_sec, ttft_ms, itl_ms.
3. cost_per_1m_tok_usd = (instance $/hr) / (tokens_per_sec \* 3600 / 1e6), using
   `aoe-capacity` for the region's price.
4. Report a table with exactly the fields a registry `instances[]` row records.
