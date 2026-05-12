# Agent Sandbox — Reference Agent

A minimal Python agent that exercises the `infra/agent-sandbox/` blueprint end-to-end. Runs inside a gVisor-isolated [agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) pod, calls Amazon Bedrock for content, executes model-generated code, and exercises both enforcement layers of the Cilium network policy.

Use this as a working example to build your own agent against the blueprint, or as a conformance check after deploying the infra.

## Prerequisites

- The `infra/agent-sandbox/` blueprint installed: run `../../../infra/agent-sandbox/install.sh` from a clone of the repo.
- An IAM role with `bedrock:InvokeModel` for the target Claude model and an IRSA trust policy that allows the cluster's OIDC provider for `system:serviceaccount:agent-sandboxes:sandbox-agent-sa`. See `../../../infra/agent-sandbox/manifests/iam-bedrock-trust-policy.template.json` and `iam-bedrock-permissions.template.json` for starting points.
- `kubectl` configured against the cluster (`aws eks update-kubeconfig --name agent-sandbox --region us-east-1`).

## What the agent does

Five steps, each designed to exercise a distinct part of the blueprint:

| Step | Action | Demonstrates |
|---|---|---|
| 1 | `pip install boto3` from PyPI | Cilium FQDN allowlist permits `pypi.org` + `files.pythonhosted.org` |
| 2 | Call Bedrock Claude Sonnet | IRSA credential path works through gVisor; FQDN allowlist permits `bedrock-runtime.*.amazonaws.com` + `sts.*.amazonaws.com` |
| 3 | Execute the model-generated Python snippet inside the sandbox | Sandbox can run code; syscalls flow through gVisor's Sentry userspace kernel |
| 4 | HTTP GET to a non-allowlisted FQDN | Cilium DNS proxy returns an empty answer; Python surfaces a resolution failure |
| 5 | Raw TCP connect to a non-allowlisted IP (8.8.8.8:443) | Cilium L3/L4 policy drops the SYN packet; produces a DROPPED flow in Hubble |

Expected console output:

```
Step 1 (PyPI):            PASS
Step 2 (Bedrock):         PASS
Step 3 (snippet exec):    PASS
Step 4 (FQDN block):      BLOCKED — at DNS proxy
Step 5 (IP block):        BLOCKED — at L3/L4
```

## Two enforcement layers, two observability surfaces

Cilium enforces network policy at two distinct layers, and they show up differently in Hubble:

**FQDN enforcement (Step 4)** — Cilium's DNS proxy intercepts DNS queries and filters the response. When a FQDN isn't on the allowlist, the proxy returns an empty answer. The pod sees a hostname resolution failure, never attempts a TCP connection, and produces no packet-level event to visualize. Step 4 does NOT render as a red flow in Hubble's default Service Map view. To observe the FQDN verdict directly:

```bash
# Run while the agent is executing Step 4
kubectl -n kube-system exec ds/cilium -c cilium-agent -- \
    cilium monitor --type l7 | grep "DNS proxy"
```

You'll see lines like `verdict Forwarded DNS proxy: blocked-example.example.com. A TTL: 4294967295 Answer: ''` — the empty `Answer: ''` is how Cilium denies FQDN egress.

Alternatively, add a positive filter for the FQDN in the Hubble UI filter bar (e.g., type `blocked-example.example.com`) to un-blacklist DNS events for that specific domain.

**L3/L4 enforcement (Step 5)** — Cilium's eBPF policy drops packets at the socket/network layer. When a pod attempts to connect to a non-allowlisted IP, the SYN is silently discarded. This IS visible in Hubble's default Service Map as a red DROPPED flow to `8.8.8.8:443`. No filter tuning required.

This distinction matters because adopters who only look at Hubble's default view will see Step 5 but miss Step 4. Both are doing the same work (blocking unauthorized egress); they just show up in different places.

## Running the agent

### Interactive run (one-off)

```bash
# 1. Annotate the ServiceAccount with the IAM role for IRSA (one time
#    per cluster — the annotation persists across pod rebuilds).
kubectl create serviceaccount sandbox-agent-sa -n agent-sandboxes --dry-run=client -o yaml | kubectl apply -f -
kubectl annotate serviceaccount sandbox-agent-sa -n agent-sandboxes \
    eks.amazonaws.com/role-arn=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    --overwrite

# 2. Install the agent script into a ConfigMap the Sandbox mounts.
kubectl -n agent-sandboxes create configmap sandbox-agent-script \
    --from-file=agent.py=./agent.py

# 3. Create the Sandbox.
kubectl apply -f ../../../infra/agent-sandbox/manifests/sandbox-agent.yaml

# 4. Wait for Ready, then run the agent.
kubectl -n agent-sandboxes wait --for=condition=Ready pod/sandbox-agent --timeout=300s
kubectl exec -n agent-sandboxes sandbox-agent -c agent-runtime -- python /workspace/agent.py
```

Open the Hubble UI in a second terminal to watch egress decisions live:

```bash
kubectl port-forward -n kube-system svc/hubble-ui 12000:80
# then open http://localhost:12000 and filter to namespace=agent-sandboxes
```

### Automated conformance run

`conformance.sh` (in the infra directory, `../../../infra/agent-sandbox/conformance.sh`) wraps the interactive steps above, executes the agent, and asserts the expected PASS/BLOCKED markers appear. Exits 0 on success, 1 on any failure. Useful after a blueprint install or as a regression check.

```bash
cd ../../../infra/agent-sandbox
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    ./conformance.sh
```

The script registers its own cleanup trap — by default it leaves the Sandbox pod + ConfigMap in place so repeat conformance runs are fast (no re-provisioning gVisor nodes). Pass `CLEANUP=1` to remove the sandbox resources on exit:

```bash
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
CLEANUP=1 \
    ./conformance.sh
```

IAM role + IRSA annotation are retained in both modes.

## Files

| File | Purpose |
|---|---|
| `agent.py` | The reference agent. Mounted into the Sandbox via ConfigMap; runs via `kubectl exec`. |
| `README.md` | This file. |

The Sandbox resource itself (`sandbox-agent.yaml`) and the KRO composite variant (`agent-sandbox-instance.yaml`) live under `../../../infra/agent-sandbox/manifests/` alongside the rest of the blueprint manifests. The automated conformance test (`conformance.sh`) lives under `../../../infra/agent-sandbox/` to match the repo's shell-script-under-infra convention.

## Adapting the agent

To use this pattern for your own agent:

1. Copy `agent.py` as a starting point — the boilerplate around user-site-packages import, `HOME=/workspace` handling, and the `try_egress` / `try_ip_egress` helpers all carry over.
2. Update the FQDN allowlist at `../../../infra/agent-sandbox/manifests/ciliumnetworkpolicy-sandbox-llm.yaml` to cover your agent's outbound domains.
3. If your agent needs different IAM permissions, update the IAM role (templates at `../../../infra/agent-sandbox/manifests/iam-bedrock-*.template.json`).
4. Mount your agent code into a Sandbox the same way this one does — via a ConfigMap referenced in `sandbox-agent.yaml`.

## Troubleshooting

- **`PASS: boto3 installed` missing** — Cilium policy isn't permitting PyPI. Check `kubectl -n agent-sandboxes get ciliumnetworkpolicy sandbox-llm-allowlist` is `Valid=True` and the `toFQDNs` list includes `pypi.org` + `files.pythonhosted.org`.
- **Bedrock `AccessDeniedException`** — IRSA annotation is missing or the IAM role lacks `bedrock:InvokeModel`. Check with `kubectl get sa sandbox-agent-sa -n agent-sandboxes -o yaml | grep role-arn` and confirm the role's trust policy allows the cluster's OIDC provider for `system:serviceaccount:agent-sandboxes:sandbox-agent-sa`.
- **Bedrock `ResourceNotFoundException` with "Legacy" in the message** — the target model hasn't been invoked in 30+ days. Either invoke it once manually from the AWS console or update `BEDROCK_MODEL_ID` in `agent.py` to a currently-active model.
- **Step 4 doesn't show `BLOCKED: ... No address associated with hostname`** — the agent reached the FQDN somehow. Check that the target FQDN (default: `blocked-example.example.com`) is NOT in the `toFQDNs` allowlist in `ciliumnetworkpolicy-sandbox-llm.yaml`, and that the CNP is `Valid=True`.
- **Step 5 doesn't show `BLOCKED: ... L3/L4 policy drop`** — the connect succeeded or errored differently. Check that the sandbox pod has labels matching the CNP's `endpointSelector` (`egress-tier: sandbox`), and that Cilium policy enforcement is active: `kubectl -n kube-system exec ds/cilium -c cilium-agent -- cilium status | grep Enforcement` should show `Default` or stricter.
- **Neither Step 4 nor Step 5 is visible in Hubble UI** — Step 4's invisibility is expected (FQDN enforcement happens at DNS proxy layer, not L3/L4). Step 5's DROPPED flow to `8.8.8.8:443` should always render in the default Hubble Service Map. If it doesn't, verify hubble-relay is connected to all peer agents: `kubectl -n kube-system logs -l k8s-app=hubble-relay --tail=20 | grep -E "Connected|No connection"`.
