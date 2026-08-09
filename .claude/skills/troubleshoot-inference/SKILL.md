---
name: troubleshoot-inference
description: Use when a model deployment on EKS is failing or degraded (pod CrashLoopBackOff, OOM, CUDA errors, Karpenter not scaling, HF token / gated repo errors). Diagnoses and fixes common vLLM-on-EKS failures.
---

# Troubleshoot Inference on EKS

Diagnose and fix the common failure modes seen deploying vLLM/Dynamo on EKS.

## Steps

1. **Get the symptom.** Read pod state and recent logs:

   kubectl get pods -l app=<name>
   kubectl describe pod <pod>
   kubectl logs <pod> --tail=200

2. **Match the failure:**
   - **`Pending`, no node** -> Karpenter has not scaled. Check the NodePool matches the
     blueprint's `node.kubernetes.io/instance-type`, and that the region has capacity
     (`aoe-capacity find <instance_type>`). Capacity may be zero in this region; move to
     a region the scanner ranks higher.
   - **`OOMKilled` / CUDA OOM in logs** -> the model does not fit. Increase
     `--tensor-parallel-size` (needs more GPUs / a bigger instance) or lower
     `--max-model-len`. Regenerate with `aoe-blueprint gen`.
   - **`CUDA error` / driver mismatch** -> the node's NVIDIA driver is older than the
     container's CUDA. Confirm the GPU AMI / device-plugin versions match the image.
   - **401 / gated repo pulling weights** -> the `hf-token` secret is missing or the
     token lacks access. Recreate it:
     `kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN`.
   - **`ValueError: architecture not recognized`** -> vLLM version predates the model
     arch. Pin a newer `container_image` in the registry entry.

3. **Verify the fix.** Re-apply, then re-run the deploy-model verification curl. Report
   what the root cause was and what changed.
