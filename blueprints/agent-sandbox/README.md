# Agent Sandbox — Reference Blueprint

## Table of Contents

- [Overview](#overview)
- [What's in this blueprint](#whats-in-this-blueprint)
- [What the agent does](#what-the-agent-does)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Apply the SandboxClaim and reference agent](#apply-the-sandboxclaim-and-reference-agent)
  - [Run the agent interactively](#run-the-agent-interactively)
  - [Automated conformance run](#automated-conformance-run)
- [KRO composition path](#kro-composition-path)
- [Egress enforcement](#egress-enforcement)
- [Two enforcement layers, two observability surfaces](#two-enforcement-layers-two-observability-surfaces)
- [Adapting the agent](#adapting-the-agent)
- [Files in this directory](#files-in-this-directory)
- [Troubleshooting](#troubleshooting)

## Overview

A complete reference implementation that runs an AI agent inside a gVisor-isolated Sandbox on Amazon EKS, with FQDN egress enforcement and end-to-end conformance. Layered on top of the [agent-sandbox infrastructure](../../infra/agent-sandbox/) which provides the platform primitives (controller + RuntimeClass + basic SandboxTemplates).

For a smaller starting point that exercises the platform without the agent-specific assumptions, see the [basic blueprint](basic/) — a minimal SandboxClaim against the platform's basic templates with `nginx:alpine` as the default workload.

This blueprint exists for two reasons:

- A working example to copy when building your own agent against the agent-sandbox infrastructure.
- A conformance check that validates the full chain end-to-end after deploying the platform.

## What's in this blueprint

| Subdirectory / file | Purpose |
|---|---|
| `agent.py` | The reference agent (Python) — five steps exercising FQDN + L3/L4 enforcement and gVisor's syscall boundary. |
| `manifests/sandbox-agent.yaml` | The reference SandboxClaim + ServiceAccount + agent-script ConfigMap. The claim's `sandboxTemplateRef.name` is patched at apply time (`sandbox-agent-gvisor` on Standard EKS, `sandbox-agent-runc` on Auto Mode). |
| `manifests/sandbox-agent-runc.yaml` | Agent-shaped SandboxTemplate for the runc tier — adds `python:3.12-slim`, agent-script ConfigMap mount, Bedrock env vars, and `sandbox-agent-sa` IRSA-bound ServiceAccount on top of the `sandbox-runc` shape. |
| `manifests/sandbox-agent-gvisor.yaml` | Same as `sandbox-agent-runc` with `runtimeClassName: gvisor` and the gVisor NodePool toleration. Standard EKS only. |
| `manifests/kro/rgd.yaml` | KRO `ResourceGraphDefinition` exposing a single `AgentSandbox` CRD wrapping the same workload shape. Optional. |
| `manifests/kro/instance.yaml` | Sample `AgentSandbox` instance used by the KRO composition path. |
| `egress/` | Egress enforcement example — auto-detects compute mode and applies Cilium CNPs (Standard EKS) or native ANPs (Auto Mode), plus IRSA bootstrap for Bedrock access. See [`egress/README.md`](egress/README.md). |
| `basic/` | Smallest viable Sandbox deployment — a SandboxClaim against the platform's basic templates (nginx:alpine workload, no IRSA, no agent script). Use as the first tier of testing or as a starting point for non-agent workloads. See [`basic/README.md`](basic/README.md). |
| `conformance.sh` | End-to-end test runner: claims the right agent template, applies the agent script, executes the agent, asserts PASS/BLOCKED markers. |

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

- The [agent-sandbox infrastructure](../../infra/agent-sandbox/) deployed: from a clone of the repo, `cd infra/agent-sandbox && ./install.sh`. This sets up the cluster, the SIG-Apps controller, KRO (optional), Cilium (optional), and the platform manifests (RuntimeClass, SandboxTemplates, namespace, gVisor Karpenter NodePool).
- The [egress example](egress/) applied — auto-detects compute mode and applies Cilium CNPs (Standard EKS) or native ANPs (Auto Mode), and provisions the Bedrock IRSA role used by the reference agent.
- An IAM role with `bedrock:InvokeModel` permission for the target Claude model, plus an IRSA trust policy allowing the cluster's OIDC provider for `system:serviceaccount:agent-sandboxes:sandbox-agent-sa`. The egress example's `irsa` phase provisions this automatically; templates at [`manifests/iam/bedrock-trust-policy.template.json`](manifests/iam/bedrock-trust-policy.template.json) and [`manifests/iam/bedrock-permissions.template.json`](manifests/iam/bedrock-permissions.template.json) for hand-rolled setups.
- `kubectl` configured against the cluster (`aws eks update-kubeconfig --name agent-sandbox --region <region>`).

## Quick Start

The recommended path is `conformance.sh` — it auto-detects the cluster's compute mode, claims the right SandboxTemplate, applies the agent ConfigMap, runs the agent, and validates the markers. The interactive path below is for first-time exploration or step-by-step debugging.

### Apply the SandboxClaim and reference agent

The reference SandboxClaim (`manifests/sandbox-agent.yaml`) targets one of the agent-shaped SandboxTemplates that ship with this blueprint. Apply both templates first (one of them gets claimed depending on compute mode):

```bash
kubectl apply -f manifests/sandbox-agent-runc.yaml
# Standard EKS only — gVisor isn't available on Auto Mode:
kubectl apply -f manifests/sandbox-agent-gvisor.yaml
```

The SandboxClaim carries a `__SANDBOX_TEMPLATE__` placeholder substituted at apply time. To apply by hand:

```bash
SANDBOX_TEMPLATE=sandbox-agent-gvisor   # or sandbox-agent-runc for Auto Mode
sed "s|__SANDBOX_TEMPLATE__|$SANDBOX_TEMPLATE|g" manifests/sandbox-agent.yaml \
    | kubectl apply -f -
```

`conformance.sh` does both the template apply and the substitution automatically based on the cluster's compute mode.

### Run the agent interactively

```bash
# 1. Annotate the ServiceAccount with your Bedrock IAM role ARN.
kubectl annotate serviceaccount sandbox-agent-sa -n agent-sandboxes \
    "eks.amazonaws.com/role-arn=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel>" \
    --overwrite

# 2. Load this agent.py into the ConfigMap the SandboxTemplate mounts.
kubectl -n agent-sandboxes create configmap sandbox-agent-script \
    --from-file=agent.py=./agent.py \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. Wait for Ready, then run the agent.
kubectl -n agent-sandboxes wait --for=condition=Ready pod/sandbox-agent --timeout=120s
kubectl exec -n agent-sandboxes sandbox-agent -c agent-runtime -- python /workspace/agent.py
```

Expected output is the 5-step sequence with PASS / PASS / PASS / BLOCKED / BLOCKED markers.

### Automated conformance run

```bash
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    ./conformance.sh
```

`conformance.sh` resolves region from the infra's `terraform/blueprint.tfvars` (with `AWS_REGION` env override), auto-detects whether the cluster is Standard EKS or Auto Mode, and validates the expected CNP/ANP resources accordingly. Exits 0 on success, 1 on any failure.

## KRO composition path

If your team prefers a single user-facing CRD over the SIG-Apps `Sandbox` API directly, the KRO `ResourceGraphDefinition` at `manifests/kro/rgd.yaml` exposes an `AgentSandbox` CRD that wraps the same workload shape. Apply the RGD and then create instances against it:

```bash
kubectl apply -f manifests/kro/rgd.yaml
# Edit manifests/kro/instance.yaml with your runtime class + IAM role ARN, then:
kubectl apply -f manifests/kro/instance.yaml
```

KRO is optional — the SandboxClaim path produces equivalent running pods. If your infra was deployed with `enable_kro = false`, this composition path is unavailable.

## Egress enforcement

The [`egress/`](egress/) subdirectory ships a mode-aware example that auto-detects compute mode and applies the right enforcement layer:

- **Standard EKS** → Cilium `CiliumClusterwideNetworkPolicy` + `CiliumNetworkPolicy` (Cilium itself is deployed by the base infra when `enable_cilium = true`).
- **EKS Auto Mode** → native `ClusterNetworkPolicy` + `ApplicationNetworkPolicy` (DNS-based, enforced by the VPC CNI Network Policy Controller).

```bash
cd egress
./install.sh                # Auto-detects mode + applies policies + provisions IRSA
```

The `install.sh` also provisions the Bedrock IRSA role used by the reference agent (idempotent; updates the trust policy on cluster recreation so OIDC drift doesn't break re-runs). The role ARN is echoed at the end for use with `conformance.sh`. Run `./install.sh irsa` to refresh the role only without re-running policy installation. See [`egress/README.md`](egress/README.md) for allowlist-template usage and migration paths between the two enforcement backends.

## Two enforcement layers, two observability surfaces

The reference agent's Step 4 and Step 5 exercise two distinct enforcement contracts, each with a different observability surface:

### Step 4 — FQDN enforcement at the DNS proxy

Cilium's `toFQDNs` and native `ApplicationNetworkPolicy`'s `domainNames` both enforce at the DNS layer. When the pod queries a non-allowlisted FQDN, the DNS proxy returns an empty answer and the pod sees `[Errno -5] No address associated with hostname`. The pod never attempts a TCP connection, so **no L3/L4 flow is generated**.

**Observability path**: DNS proxy logs, not flow graphs.

```bash
# Standard EKS (Cilium):
CILIUM_POD=$(kubectl -n kube-system get pods -l k8s-app=cilium -o jsonpath='{.items[0].metadata.name}')
kubectl -n kube-system exec $CILIUM_POD -c cilium-agent -- cilium monitor --type l7 2>&1 | grep "DNS proxy"

# Auto Mode (VPC CNI): DNS verdicts appear in the Network Policy Agent logs
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
2. Update the FQDN allowlist to cover your agent's outbound domains. For Standard EKS (Cilium), edit [`egress/manifests/cilium/ciliumnetworkpolicy-sandbox-llm.yaml`](egress/manifests/cilium/ciliumnetworkpolicy-sandbox-llm.yaml). For Auto Mode (ANP), edit [`egress/manifests/anp/applicationnetworkpolicy-sandbox-llm.yaml`](egress/manifests/anp/applicationnetworkpolicy-sandbox-llm.yaml).
3. If your agent needs different IAM permissions, update the IAM role (templates at [`manifests/iam/bedrock-trust-policy.template.json`](manifests/iam/bedrock-trust-policy.template.json) and [`manifests/iam/bedrock-permissions.template.json`](manifests/iam/bedrock-permissions.template.json)).
4. Mount your agent code into a Sandbox the same way this one does — via a ConfigMap referenced in the `Sandbox` spec.

For larger agents where a ConfigMap mount is impractical, bake `agent.py` into a container image and reference it in `Sandbox.spec.podTemplate.spec.containers[].image` instead. Keep the `readOnlyRootFilesystem`, `runAsNonRoot`, `capabilities.drop: [ALL]`, and writable-workspace patterns from [`manifests/sandbox-agent.yaml`](manifests/sandbox-agent.yaml).

## Files in this directory

| File / subdirectory | Purpose |
|------|---------|
| `agent.py` | The reference agent — 5 steps demonstrating FQDN + L3/L4 enforcement |
| `conformance.sh` | Automated end-to-end test — applies the SandboxClaim, runs the agent, asserts PASS/BLOCKED markers |
| `manifests/sandbox-agent.yaml` | SandboxClaim + ServiceAccount + agent-script ConfigMap |
| `manifests/sandbox-agent-runc.yaml` | Agent-shaped SandboxTemplate for the runc tier |
| `manifests/sandbox-agent-gvisor.yaml` | Agent-shaped SandboxTemplate for the gVisor tier |
| `manifests/kro/{rgd,instance}.yaml` | KRO composition path — optional |
| `basic/` | Smallest viable Sandbox deployment — see [`basic/README.md`](basic/README.md) |
| `egress/` | Egress enforcement example (mode-aware, Cilium + ANP) — see [`egress/README.md`](egress/README.md) |
| `README.md` | This file |

The platform primitives this blueprint depends on (RuntimeClass, basic SandboxTemplates, namespace, gVisor Karpenter NodePool, IAM templates) live in the [parent infra](../../infra/agent-sandbox/manifests/).

## Troubleshooting

### `AccessDenied: AssumeRoleWithWebIdentity` in Step 2

The IAM role's trust policy subject doesn't match the ServiceAccount path. Verify:

```bash
aws iam get-role --role-name <role-name> --query 'Role.AssumeRolePolicyDocument'
```

The condition must include `system:serviceaccount:agent-sandboxes:sandbox-agent-sa` (and `:composed-sandbox` if you're using the KRO composition path). Update via `aws iam update-assume-role-policy`.

### Step 4 passes instead of BLOCKS

The FQDN-deny policy isn't enforcing. Most common causes:

- Auto Mode (native ANP): the Network Policy Controller isn't enabled. Check for the `amazon-vpc-cni` ConfigMap in `kube-system` with `enable-network-policy-controller: "true"`. Apply via [`egress/manifests/anp/network-policy-controller-enable.yaml`](egress/manifests/anp/network-policy-controller-enable.yaml) or re-run the egress example's install.
- Standard EKS (Cilium): Cilium isn't installed (set `enable_cilium = true` in the parent infra's `terraform/blueprint.tfvars`), or hubble-relay peer list is stale after Karpenter node cycles. Run `kubectl rollout restart deployment/hubble-relay -n kube-system` if flows look frozen.

### Step 5 passes instead of BLOCKS

The L3/L4 policy isn't in place. The default sandbox allowlist enforces default-deny for destinations not explicitly listed — verify the policy exists:

```bash
# Standard EKS (Cilium):
kubectl get ciliumnetworkpolicy -n agent-sandboxes sandbox-llm-allowlist

# Auto Mode (ANP):
kubectl get applicationnetworkpolicy -n agent-sandboxes sandbox-llm-allowlist
```

### Agent output is empty (silent `kubectl exec`)

The container's `cp /config/agent.py /workspace/agent.py` happens once at pod start. If the ConfigMap was updated after the pod was Ready, the workspace has old content. Recreate the pod:

```bash
kubectl delete pod sandbox-agent -n agent-sandboxes
kubectl -n agent-sandboxes wait --for=condition=Ready pod/sandbox-agent --timeout=120s
```

This is documented in detail in the parent infra's [Troubleshooting section](../../infra/agent-sandbox/README.md#troubleshooting).
