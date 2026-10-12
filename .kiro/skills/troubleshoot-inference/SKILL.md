---
name: troubleshoot-inference
description: Diagnose and fix a failing or degraded model deployment on EKS (CrashLoopBackOff, OOM, CUDA errors, Karpenter not scaling, HF token / gated repo errors). Use when a vLLM/Dynamo deployment on EKS is not healthy.
---

# Troubleshoot Inference on EKS

When a deployment is failing or degraded:

1. `kubectl get pods -l app=<name>`, `describe`, and `logs --tail=200`.
2. Match the failure: Pending/no node -> Karpenter/capacity (check NodePool selector and
   `aoe-capacity`); OOMKilled -> raise `--tensor-parallel-size` or lower `--max-model-len`;
   CUDA driver mismatch -> align GPU AMI/device-plugin with image; 401 gated repo ->
   recreate `hf-token` secret; unknown architecture -> pin a newer `container_image`.
3. Re-apply and re-run the deploy verification curl. Report root cause and fix.
