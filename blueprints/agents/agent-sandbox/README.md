# Agent Sandbox — Reference Agent Blueprint

## Table of Contents

- [Overview](#overview)
- [What the agent does](#what-the-agent-does)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Interactive run](#interactive-run)
  - [Automated conformance run](#automated-conformance-run)
- [Two Enforcement Layers, Two Observability Surfaces](#two-enforcement-layers-two-observability-surfaces)
- [Adapting the agent](#adapting-the-agent)
- [Files in this directory](#files-in-this-directory)
- [Troubleshooting](#troubleshooting)

## Overview

A minimal Python agent that demonstrates a secure agent workload pattern on Amazon EKS using the [agent-sandbox solution](../../../infra/solutions/agent-sandbox/). The agent runs inside a gVisor-isolated sandbox, authenticates to AWS via IRSA, calls Amazon Bedrock for content, executes model-generated code inside the Sentry boundary, and exercises both enforcement layers of the egress network policy.

Use this as a working example to build your own agent against the solution, or as a conformance check after deploying the infrastructure.

## What the agent does

The agent (`agent.py`) walks five steps and prints PASS/BLOCKED markers after each:

| Step | Action | Expected outcome | Exercises |
|------|--------|-------------------|-----------|
| 1 | `pip install boto3` from PyPI | PASS | FQDN allowlist (allow `pypi.org`) |
| 2 | Call Amazon Bedrock Claude for a code snippet | PASS | FQDN allowlist (allow `bedrock-runtime`) + IRSA |
| 3 | Execute the model-generated snippet inside the sandbox | PASS | gVisor Sentry syscall boundary |
| 4 | Attempt egress to a non-allowlisted FQDN | BLOCKED (DNS resolution failure) | Cilium/ANP DNS proxy enforcement |
| 5 | Attempt raw TCP connect to a non-allowlisted IP | BLOCKED (connection timeout) | Cilium/ANP L3/L4 enforcement |

Step 4 proves the FQDN-layer contract; Step 5 proves the L3/L4 contract. Both blocks are expected — a PASS in Step 4 or Step 5 indicates the policy isn't enforcing.

## Prerequisites

- The [agent-sandbox solution](../../../infra/solutions/agent-sandbox/) installed: run `../../../infra/solutions/agent-sandbox/install.sh` from a clone of the repo. See its README for solution-level steps (cluster provisioning, manifest application, egress enforcement).
- One of the egress examples applied ([agent-egress-chained](../../../infra/solutions/agent-sandbox/examples/agent-egress-chained/) for Standard EKS, [agent-egress-native](../../../infra/solutions/agent-sandbox/examples/agent-egress-native/) for EKS Auto Mode).
- An IAM role with `bedrock:InvokeModel` permission for the target Claude model, plus an IRSA trust policy allowing the cluster's OIDC provider for `system:serviceaccount:agent-sandboxes:sandbox-agent-sa`. See [`iam-bedrock-trust-policy.template.json`](../../../infra/solutions/agent-sandbox/manifests/iam-bedrock-trust-policy.template.json) and [`iam-bedrock-permissions.template.json`](../../../infra/solutions/agent-sandbox/manifests/iam-bedrock-permissions.template.json) for starting points.
- `kubectl` configured against the cluster (`aws eks update-kubeconfig --name agent-sandbox --region <region>`).

## Quick Start

### Interactive run

Walk through the agent step-by-step with direct kubectl commands. Useful when first exploring the solution or debugging a specific step.

```bash
# 1. Annotate the ServiceAccount with your Bedrock IAM role ARN.
kubectl annotate serviceaccount sandbox-agent-sa -n agent-sandboxes \
    "eks.amazonaws.com/role-arn=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel>" \
    --overwrite

# 2. Load this agent.py into the ConfigMap the Sandbox mounts.
kubectl -n agent-sandboxes create configmap sandbox-agent-script \
    --from-file=agent.py=./agent.py \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. Create the Sandbox.
kubectl apply -f ../../../infra/solutions/agent-sandbox/manifests/sandbox-agent.yaml

# 4. Wait for Ready, then run the agent.
kubectl -n agent-sandboxes wait --for=condition=Ready pod/sandbox-agent --timeout=120s
kubectl exec -n agent-sandboxes sandbox-agent -c agent-runtime -- python /workspace/agent.py
```

Expected output is the 5-step sequence with PASS / PASS / PASS / BLOCKED / BLOCKED markers.

### Automated conformance run

`conformance.sh` wraps the interactive steps above, executes the agent, and asserts the expected markers appear. Exits 0 on success, 1 on any failure. Useful after a solution install or as a regression check.

```bash
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    ./conformance.sh
```

The script auto-detects whether chained or native egress is installed and validates the expected CNP/ANP resources accordingly.

## Two Enforcement Layers, Two Observability Surfaces

The reference agent's Step 4 and Step 5 exercise two distinct enforcement contracts, each with a different observability surface:

### Step 4 — FQDN enforcement at the DNS proxy

Cilium's `toFQDNs` and native `ApplicationNetworkPolicy`'s `domainNames` both enforce at the DNS layer. When the pod queries a non-allowlisted FQDN, the DNS proxy returns an empty answer and the pod sees `[Errno -5] No address associated with hostname`. The pod never attempts a TCP connection, so **no L3/L4 flow is generated**.

**Observability path**: DNS proxy logs, not flow graphs.

```bash
# Chained (Cilium):
CILIUM_POD=$(kubectl -n kube-system get pods -l k8s-app=cilium -o jsonpath='{.items[0].metadata.name}')
kubectl -n kube-system exec $CILIUM_POD -c cilium-agent -- cilium monitor --type l7 2>&1 | grep "DNS proxy"

# Native (VPC CNI): DNS verdicts appear in the Network Policy Agent logs
kubectl logs -n kube-system -l app=aws-node -c aws-network-policy-agent | grep -i "dns"
```

Hubble UI's default Service Map filters blacklist DNS events, which is why a denied FQDN doesn't render as a red flow in the default view. This is correct behavior — the Service Map is aggregated topology, not a DNS log.

### Step 5 — L3/L4 enforcement at eBPF

When the pod attempts a raw TCP connection to a non-allowlisted IP (bypassing DNS), the network policy's L3/L4 rules drop the SYN packet. The pod sees a connection timeout.

**Observability path**: default Hubble UI Service Map shows a red DROPPED flow. No special filter tuning needed.

This is the "visible denial" that the reference agent is structured to produce — Step 4 alone wouldn't render anything in the default observability surface.

## Adapting the agent

To build your own agent on this pattern:

1. Copy `agent.py` as a starting point — the boilerplate around user-site-packages import, `HOME=/workspace` handling, and the `try_egress` / `try_ip_egress` helpers all carry over.
2. Update the FQDN allowlist to cover your agent's outbound domains. For the chained path, edit [`ciliumnetworkpolicy-sandbox-llm.yaml`](../../../infra/solutions/agent-sandbox/examples/agent-egress-chained/manifests/ciliumnetworkpolicy-sandbox-llm.yaml). For the native path, edit [`applicationnetworkpolicy-sandbox-llm.yaml`](../../../infra/solutions/agent-sandbox/examples/agent-egress-native/manifests/applicationnetworkpolicy-sandbox-llm.yaml).
3. If your agent needs different IAM permissions, update the IAM role (templates at [`iam-bedrock-trust-policy.template.json`](../../../infra/solutions/agent-sandbox/manifests/iam-bedrock-trust-policy.template.json) and [`iam-bedrock-permissions.template.json`](../../../infra/solutions/agent-sandbox/manifests/iam-bedrock-permissions.template.json)).
4. Mount your agent code into a Sandbox the same way this one does — via a ConfigMap referenced in the `Sandbox` spec.

For larger agents where a ConfigMap mount is impractical, bake `agent.py` into a container image and reference it in `Sandbox.spec.podTemplate.spec.containers[].image` instead. Keep the `readOnlyRootFilesystem`, `runAsNonRoot`, `capabilities.drop: [ALL]`, and writable-workspace patterns from [`sandbox-agent.yaml`](../../../infra/solutions/agent-sandbox/manifests/sandbox-agent.yaml).

## Files in this directory

| File | Purpose |
|------|---------|
| `agent.py` | The reference agent — 5 steps demonstrating FQDN + L3/L4 enforcement |
| `conformance.sh` | Automated end-to-end test — applies manifests, runs the agent, asserts PASS/BLOCKED markers |
| `README.md` | This file |

The `Sandbox` resource (`sandbox-agent.yaml`), the KRO composition variant (`agent-sandbox-instance.yaml` + `rgd-agent-sandbox.yaml`), and supporting manifests live under [`../../../infra/solutions/agent-sandbox/manifests/`](../../../infra/solutions/agent-sandbox/manifests/).

## Troubleshooting

### `AccessDenied: AssumeRoleWithWebIdentity` in Step 2

The IAM role's trust policy subject doesn't match the ServiceAccount path. Verify:

```bash
aws iam get-role --role-name <role-name> --query 'Role.AssumeRolePolicyDocument'
```

The condition must include `system:serviceaccount:agent-sandboxes:sandbox-agent-sa` (and `:composed-sandbox` if you're using the KRO composition path). Update via `aws iam update-assume-role-policy`.

### Step 4 passes instead of BLOCKS

The FQDN-deny policy isn't enforcing. Most common causes:

- Native path on Auto Mode: the Network Policy Controller isn't enabled. Check for the `amazon-vpc-cni` ConfigMap in `kube-system` with `enable-network-policy-controller: "true"`. Apply via [`network-policy-controller-enable.yaml`](../../../infra/solutions/agent-sandbox/examples/agent-egress-native/manifests/network-policy-controller-enable.yaml) or re-run the native egress example's install.
- Chained path on Standard EKS: Cilium isn't installed, or hubble-relay peer list is stale after Karpenter node cycles. Run `kubectl rollout restart deployment/hubble-relay -n kube-system` if flows look frozen.

### Step 5 passes instead of BLOCKS

The L3/L4 policy isn't in place. The default sandbox allowlist enforces default-deny for destinations not explicitly listed — verify the policy exists:

```bash
# Chained:
kubectl get ciliumnetworkpolicy -n agent-sandboxes sandbox-llm-allowlist

# Native:
kubectl get applicationnetworkpolicy -n agent-sandboxes sandbox-llm-allowlist
```

### Agent output is empty (silent `kubectl exec`)

The container's `cp /config/agent.py /workspace/agent.py` happens once at pod start. If the ConfigMap was updated after the pod was Ready, the workspace has old content. Recreate the pod:

```bash
kubectl delete pod sandbox-agent -n agent-sandboxes
kubectl -n agent-sandboxes wait --for=condition=Ready pod/sandbox-agent --timeout=120s
```

This is documented in detail in the parent solution's [Troubleshooting section](../../../infra/solutions/agent-sandbox/README.md#troubleshooting).
