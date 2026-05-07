# Agent Sandbox — Reference Agent

A minimal Python agent that demonstrates the `infra/agent-sandbox/` blueprint end-to-end. Runs inside a gVisor-isolated [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pod, calls Amazon Bedrock for content, executes model-generated code, and exercises the FQDN egress allowlist both for permitted and blocked domains.

Use this as a working example to build your own agent against the blueprint, or as a conformance check after deploying the infra.

## Prerequisites

- The `infra/agent-sandbox/` blueprint installed: run `../../../infra/agent-sandbox/install.sh` from a clone of the repo.
- An IAM role with `bedrock:InvokeModel` for the target Claude model and a trust policy that allows EKS Pod Identity. See `../../../infra/agent-sandbox/manifests/iam-bedrock-trust-policy.template.json` and `iam-bedrock-permissions.template.json` for starting points.
- `kubectl` configured against the cluster (`aws eks update-kubeconfig --name agent-sandbox --region us-east-1`).

## What the agent does

Four steps, each designed to exercise a distinct part of the blueprint:

| Step | Action | Demonstrates |
|---|---|---|
| 1 | `pip install boto3` from PyPI | Cilium FQDN allowlist permits `pypi.org` + `files.pythonhosted.org` |
| 2 | Call Bedrock Claude Sonnet | IRSA credential path works through gVisor; FQDN allowlist permits `bedrock-runtime.*.amazonaws.com` + `sts.*.amazonaws.com` |
| 3 | Execute the model-generated Python snippet inside the sandbox | Sandbox can run code; syscalls flow through gVisor's Sentry userspace kernel |
| 4 | HTTP GET to a non-allowlisted domain | Cilium policy denies the flow; Hubble shows a DROP event |

Expected console output:

```
Step 1 (PyPI):            PASS
Step 2 (Bedrock):         PASS
Step 3 (snippet exec):    PASS
Step 4 (blocked egress):  BLOCKED
```

## Running the agent

### Interactive run (one-off)

```bash
# 1. Create the Pod Identity association (one time per cluster).
aws eks create-pod-identity-association \
    --cluster-name agent-sandbox \
    --namespace agent-sandboxes \
    --service-account sandbox-demo-agent \
    --role-arn arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    --region us-east-1

# 2. Install the agent script into a ConfigMap the Sandbox mounts.
kubectl -n agent-sandboxes create configmap sandbox-demo-agent-script \
    --from-file=agent.py=./agent.py

# 3. Create the Sandbox.
kubectl apply -f ../../../infra/agent-sandbox/manifests/demo-agent.yaml

# 4. Wait for Ready, then run the agent.
kubectl -n agent-sandboxes wait --for=condition=Ready pod/sandbox-demo --timeout=300s
kubectl exec -n agent-sandboxes sandbox-demo -c agent-runtime -- python /workspace/agent.py
```

Open the Hubble UI in a second terminal to watch the egress decisions live:

```bash
kubectl port-forward -n kube-system svc/hubble-ui 12000:80
# then open http://localhost:12000 and filter to namespace=agent-sandboxes
```

### Automated conformance run

`conformance.sh` wraps the steps above, executes the agent, and asserts the expected PASS/BLOCKED markers appear. Exits 0 on success, 1 on any failure. Useful after a blueprint install or as a regression check.

```bash
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    ./conformance.sh
```

The script registers its own cleanup trap — the Sandbox pod and ConfigMap are removed on exit regardless of pass/fail. Pod Identity association + IAM role are retained for re-runs.

## Files

| File | Purpose |
|---|---|
| `agent.py` | The reference agent. Imported into the Sandbox via ConfigMap; runs via `kubectl exec`. |
| `conformance.sh` | Automated setup + run + assertion + cleanup. |
| `README.md` | This file. |

The Sandbox resource itself (`demo-agent.yaml`) and the KRO composite variant (`agent-sandbox-demo-instance.yaml`) live under `../../../infra/agent-sandbox/manifests/` alongside the rest of the blueprint manifests.

## Adapting the agent

To use this pattern for your own agent:

1. Copy `agent.py` as a starting point — the boilerplate around user-site-packages import, `HOME=/workspace` handling, and the `try_egress` helper all carry over.
2. Update the FQDN allowlist at `../../../infra/agent-sandbox/manifests/ciliumnetworkpolicy-sandbox-llm.yaml` to cover your agent's outbound domains.
3. If your agent needs different IAM permissions, update the IAM role (templates at `../../../infra/agent-sandbox/manifests/iam-bedrock-*.template.json`).
4. Mount your agent code into a Sandbox the same way this one does — via a ConfigMap referenced in `demo-agent.yaml`.

## Troubleshooting

- **`PASS: boto3 installed` missing** — Cilium policy isn't permitting PyPI. Check `kubectl -n agent-sandboxes get ciliumnetworkpolicy sandbox-llm-allowlist` is `Valid=True` and the `toFQDNs` list includes `pypi.org` + `files.pythonhosted.org`.
- **Bedrock `AccessDeniedException`** — Pod Identity association is missing or the IAM role lacks `bedrock:InvokeModel`. Run `aws eks list-pod-identity-associations --cluster-name agent-sandbox --namespace agent-sandboxes --service-account sandbox-demo-agent`.
- **Bedrock `ResourceNotFoundException` with "Legacy" in the message** — the target model hasn't been invoked in 30+ days. Either invoke it once manually from the AWS console or update `BEDROCK_MODEL_ID` in `agent.py` to a currently-active model.
- **`BLOCKED: https://demo-blocked.example.com/` missing** — the blocked step isn't being reached; check earlier steps for errors.
- **Step 4 unexpectedly PASSes** — the FQDN policy isn't enforcing. Confirm Cilium chaining took effect: `kubectl -n kube-system exec ds/cilium -- cilium status | grep Enforcement` should show "Policy Enforcement: Default".
