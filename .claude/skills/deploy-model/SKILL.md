---
name: deploy-model
description: Use when the user wants to deploy an open-weight model on EKS (e.g. "deploy qwen3-coder-30b", "run Kimi K3 on the cluster"). Reads the registry, checks capacity, generates a blueprint, applies it, and verifies the endpoint.
---

# Deploy Model on EKS

Deploy a registered open-weight model onto the EKS cluster and confirm it serves.

## Steps

1.  **Resolve the entry.** Look for `registry/models/<name>.yaml` matching the model the
    user named. If it exists, read it. If it does NOT exist, tell the user the model is
    unverified and offer the best-effort sizing walk-through (params -> VRAM -> instance
    count); mark anything you produce as unverified. Do not invent benchmark numbers.

2.  **Check capacity.** Read the first `instances[].type` from the entry (the verified
    instance type). Run:

        aoe-capacity find <instance_type> --regions all

    Pick the top-ranked (region, price) row. Tell the user the region and expected $/hr.

3.  **Ensure the HF token secret exists.** The blueprint reads the Hugging Face token
    from the `hf-token` secret. Verify or create it:

        kubectl get secret hf-token || \
          kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN

4.  **Generate the blueprint.** Default target is `vllm`; use `dynamo` only if the user
    asks for the Dynamo graph deployment:

        aoe-blueprint gen registry/models/<name>.yaml --target vllm -o /tmp/<name>.yaml

5.  **Apply it.**

    kubectl apply -f /tmp/<name>.yaml

6.  **Verify the endpoint.** Wait for the pod, then hit the OpenAI-compatible endpoint:

        kubectl rollout status deployment/<name> --timeout=20m
        kubectl port-forward svc/<name> 8000:8000 &
        curl -s localhost:8000/v1/chat/completions \
          -H 'Content-Type: application/json' \
          -d '{"model":"<hf_repo>","messages":[{"role":"user","content":"hello"}]}'

    A JSON completion means the deploy is live. Report the region, instance type, and a
    one-line sample of the response. If the pod is stuck, hand off to the
    troubleshoot-inference skill.
