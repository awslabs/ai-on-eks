---
name: benchmark-model
description: Use when the user wants to benchmark a deployed inference endpoint (throughput, TTFT, ITL, cost). Runs a concurrency sweep against an OpenAI-compatible endpoint on the cluster.
---

# Benchmark a Deployed Model

Measure tokens/sec, time-to-first-token (TTFT), inter-token latency (ITL), and cost per
1M tokens for a running endpoint.

## Steps

1. **Locate the endpoint.** Confirm the deployment is up and port-forward it:

   kubectl get deployment <name>
   kubectl port-forward svc/<name> 8000:8000 &

2. **Run a concurrency sweep.** Drive the endpoint at 2-3 concurrency points
   (e.g. 1, 4, 8), capturing tokens/sec, TTFT, and ITL at each. Use the same
   `inference-perf`-style harness the factory uses; point it at
   `http://localhost:8000/v1` with the model id set to the entry's `hf_repo`.

3. **Compute cost/1M tokens.** cost_per_1m_tok_usd = (instance $/hr) / (tokens_per_sec \*
   3600 / 1e6). Use the price from `aoe-capacity find <instance_type>` for the region the
   model runs in.

4. **Report.** Produce a table: concurrency, tokens_per_sec, ttft_ms, itl_ms,
   cost_per_1m_tok_usd. These are exactly the fields a `registry/models/<name>.yaml`
   `instances[]` row records — hand them to whoever updates the registry entry.
