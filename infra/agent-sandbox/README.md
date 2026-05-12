# Agent Sandbox on EKS — Blueprint

Deploys an EKS cluster that supports running AI agents inside kernel-isolated sandboxes. Built on [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) + Karpenter, with optional KRO composition.

Scope: cluster + Karpenter + sandbox controller + two runtime tiers (`standard` / `gvisor`) + a reference Python agent. **Egress enforcement ships as a separate blueprint** — pair this with one of:

- `infra/agent-egress-chained/` — VPC CNI + Cilium chaining. Available on Standard EKS today.
- `infra/agent-egress-native/` — VPC CNI + native `ApplicationNetworkPolicy`. Available on EKS Auto Mode.

A reference Python agent lives under `../../blueprints/agents/agent-sandbox/` — see that directory's README for how to run and adapt it.

## What gets installed

| Layer | Component | Source |
|---|---|---|
| Cluster | EKS v1.34 (Standard or Auto Mode) | ai-on-eks base module |
| Workers | Karpenter 1.11+ (Standard) or Auto Mode node groups | ai-on-eks base module |
| Sandbox runtime | kubernetes-sigs/agent-sandbox v0.4.3 | Blueprint overlay |
| Runtime tiers | `standard` (runc) + `gvisor` | Blueprint overlay |
| Composition (optional) | kro AgentSandbox RGD | Blueprint overlay |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ Agent workload (Python agent, background processor, etc.)    │
│ Runs inside a Sandbox (agents.x-k8s.io CRD).                 │
└──────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ SIG-Apps agent-sandbox controller                            │
│ Manages Sandbox + SandboxTemplate + SandboxClaim lifecycle.  │
└──────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ RuntimeClass selection                                       │
│   standard (runc)      │   gvisor (runsc + Sentry)           │
│   Default K8s runtime  │   Userspace syscall interception    │
│   Cold start ~1s       │   Cold start ~1.5s                  │
└──────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ Egress enforcement (separate blueprint)                      │
│   agent-egress-chained   │   agent-egress-native              │
│   (Cilium, Standard EKS) │   (VPC CNI ANP, EKS Auto Mode)     │
└──────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ EKS Node Groups                                              │
│   Karpenter-provisioned (Standard)                           │
│   Auto Mode-managed (Auto Mode)                              │
│   Managed Node Group (documented alternative for gVisor)     │
└──────────────────────────────────────────────────────────────┘
```

## Prerequisites

- AWS credentials with permissions for VPC, EKS, IAM, EC2, and (for the reference agent) `bedrock:InvokeModel` on the target Claude model.
- `terraform >=1.9`, `kubectl >=1.30`, `helm >=3.18`, `aws` CLI v2.
- An IAM role with `bedrock:InvokeModel` + IRSA trust policy allowing the cluster's OIDC provider — templates at `manifests/iam-bedrock-trust-policy.template.json` and `manifests/iam-bedrock-permissions.template.json`.

## Usage

Standard install path (Standard EKS + chained egress):

```bash
cd infra/agent-sandbox && ./install.sh            # 20-30 min
cd ../agent-egress-chained && ./install.sh        # 3-5 min
cd ../agent-sandbox && ./conformance.sh           # end-to-end validation
```

Auto Mode install path (EKS Auto Mode + native egress):

```bash
cd infra/agent-sandbox
# Uncomment `enable_eks_auto_mode = true` in terraform/blueprint.tfvars before running
./install.sh                                      # 20-30 min
cd ../agent-egress-native && ./install.sh         # ~30s
cd ../agent-sandbox && ./conformance.sh           # end-to-end validation
```

This blueprint phased by itself:

```bash
./install.sh cluster    # base EKS cluster only (20-30 min)
./install.sh sandbox    # + kubernetes-sigs/agent-sandbox controller (1-2 min)
./install.sh manifests  # + RuntimeClass, NodePool, SandboxTemplates (~30s)
./install.sh kro        # + KRO + AgentSandbox RGD (optional)
```

Each phase is idempotent. Re-run `./install.sh manifests` to re-apply after edits.

Deploy and run the reference agent (requires one of the egress blueprints installed):

```bash
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    ./conformance.sh
```

See `../../blueprints/agents/agent-sandbox/README.md` for interactive-run instructions and troubleshooting.

Destroy:

```bash
cd infra/agent-sandbox/terraform/_LOCAL
./cleanup.sh
```

## Directory layout

```
infra/agent-sandbox/
├── README.md                                     # This file
├── install.sh                                    # Phased installer (cluster | sandbox | manifests | kro | all)
├── conformance.sh                                # End-to-end conformance test (calls the reference agent)
├── terraform/
│   └── blueprint.tfvars                          # Feature flags for the base ai-on-eks module
└── manifests/
    ├── namespace.yaml                            # agent-sandboxes namespace with egress-tier label
    ├── runtimeclass-gvisor.yaml                  # gVisor RuntimeClass pointing at containerd's runsc handler
    ├── karpenter-nodepool-gvisor.yaml            # Dedicated NodePool with AL2023 + gVisor shim user-data
    ├── sandbox-template-standard.yaml            # Tier 1 SandboxTemplate (runc)
    ├── sandbox-template-gvisor.yaml              # Tier 2 SandboxTemplate (gVisor)
    ├── sandbox-agent.yaml                        # Direct Sandbox resource for the reference agent
    ├── rgd-agent-sandbox.yaml                    # KRO ResourceGraphDefinition — composes Sandbox + SA
    ├── agent-sandbox-instance.yaml               # AgentSandbox instance via the KRO RGD (optional)
    ├── kro-install.sh                            # Installs KRO on the cluster
    ├── iam-bedrock-trust-policy.template.json    # IRSA trust policy template
    └── iam-bedrock-permissions.template.json     # Bedrock permissions template
```

Reference agent lives under `blueprints/agents/agent-sandbox/` — matches the ai-on-eks convention where infra lives in `infra/` and workload patterns live in `blueprints/`.

## Threat model per tier

Each runtime tier is a weaker boundary than the one below it. Tier choice maps to a threat model, not to a "is it secure enough?" question.

| Tier | Boundary | Protects against | Does NOT protect against |
|---|---|---|---|
| standard (runc) | Linux namespaces + cgroups | Noisy neighbors, resource exhaustion, basic separation between tenant workloads | Kernel exploits, syscall abuse, privilege escalation via kernel vulnerabilities |
| gVisor (runsc + Sentry) | Userspace syscall interception | All of the above, plus most kernel-attack-surface classes. Syscalls are filtered through Sentry's userspace implementation rather than hitting the host kernel directly. | Hardware-level side channels (Spectre-class). Kata + Firecracker closes this gap. |
| Kata + Firecracker (future) | Hardware-enforced microVM (KVM) | All of the above, including hardware-level side channels. Each sandbox gets its own VM with isolated CPU state. | Not shipped in this blueprint — requires nested virtualization support which EKS Managed Node Groups do not yet provide. |

## Tier selection decision tree

1. Does your agent execute untrusted code (prompts that generate + run code, user-uploaded scripts, model-generated shell)?
   - No → `standard` tier is sufficient. Lower cold-start, simpler operational surface.
   - Yes → continue.
2. Can the agent tolerate ~50ms of syscall overhead per operation?
   - Yes → `gvisor` tier. Strong isolation without the microVM cost.
   - No → wait for the Kata + Firecracker tier, or run those workloads on dedicated EC2 bare-metal outside the cluster.
3. Is the workload multi-tenant across security boundaries (different customer tenants, regulatory domains)?
   - Yes → `gvisor` minimum; evaluate Kata + Firecracker when available.
   - No → `gvisor` is still a reasonable default for code-executing agents even in single-tenant deployments.

## Operational notes

- **gVisor cold start**: first pod on a gVisor node takes 60-90s while Karpenter bootstraps + the `runsc` shim installs via user-data. Subsequent pods on the same node are ~1.5s. Pre-warm the NodePool by applying a throwaway pod with `runtimeClassName: gvisor` ~5 minutes before you need the first real workload.
- **gVisor resource overhead**: Sentry (the userspace kernel) runs per-sandbox and consumes ~20-50MB RAM + a small CPU fraction per active syscall. Noticeable under heavy I/O workloads; negligible for typical agent workflows (LLM calls + occasional Python execution).
- **Hubble UI memory**: ~100MB per node for the relay, ~50MB for the UI pod. Ships disabled by default on the agent-sandbox blueprint (lives in `agent-egress-chained/`).
- **IRSA latency**: STS AssumeRoleWithWebIdentity adds ~100ms to the first AWS SDK call per pod. Subsequent calls use the cached credential. Trivial for agent workloads.

## MNG alternative for gVisor nodes

The default gVisor NodePool uses Karpenter for on-demand provisioning. Some customers prefer EKS Managed Node Groups (MNG) for stricter IAM boundaries or existing operational patterns. See `manifests/mng-sample/launch-template.yaml` for a working MNG launch-template user-data sample that installs the gVisor runsc shim on AL2023.

**Constraints**:
- Kata + Firecracker on MNG is blocked today by EKS's MNG CPU Options override, which removes the `nested-virtualization` flag. `gvisor` on MNG works fine because gVisor is a userspace shim, not a hardware-virt solution.
- MNG costs slightly more than Karpenter Spot for equivalent capacity and requires explicit scaling rather than demand-driven provisioning.

## Design decisions

- **Sandbox runtime is separate from egress enforcement** — this blueprint ships only the sandbox concern. Egress is its own concern with substantially different operational constraints (Cilium chaining vs VPC CNI native, Standard vs Auto Mode). Splitting them reduces the review surface and lets customers mix-and-match.
- **Raw manifest install for agent-sandbox** — upstream doesn't ship a Helm chart; this blueprint matches their install pattern rather than wrapping.
- **AL2023 over Bottlerocket for gVisor nodes** — the gVisor containerd shim install scripts upstream target AL2023 systemd. Bottlerocket would require a custom variant build.
- **IRSA rather than EKS Pod Identity for gVisor-tier workloads** — Pod Identity's credential endpoint (169.254.170.23) is served on the host network namespace. gVisor's Sentry doesn't forward that link-local route through the sandbox's virtual network stack, so Pod Identity isn't reachable from inside a gVisor sandbox. IRSA (`AWS_WEB_IDENTITY_TOKEN_FILE` + STS) routes through the regular network path, which the FQDN egress policy permits via `sts.*.amazonaws.com`.
- **kro composition is optional** — adapts `composing-sandbox-nw-policies/rgd.yaml` from upstream. Customers can use the blueprint without adopting kro.

## Known limitations

1. **gVisor shim install via Karpenter user-data adds 60-90s to first-pod-on-node latency.** Pre-warm the gVisor NodePool before a demo or benchmark run.
2. **Kata + Firecracker tier is not included.** Blocked upstream on EKS MNG nested-virt support; the self-managed path works but isn't integrated with Karpenter-provisioned capacity in this blueprint today.
3. **The reference agent exercises both FQDN and L3/L4 enforcement** (Steps 4 and 5 respectively). This depends on an egress blueprint being installed — the reference agent will exit non-zero on Steps 4/5 without one.

## Troubleshooting

- **gVisor pod stays Pending** — Karpenter may be waiting for a suitable instance. Check `kubectl get nodepools` + `kubectl describe nodeclaim`. If the pool is sized out, bump `limits.cpu/memory` in `karpenter-nodepool-gvisor.yaml`.
- **Bedrock call fails with `AccessDeniedException`** — the IRSA annotation is missing or the IAM role lacks `bedrock:InvokeModel`. Check `kubectl -n agent-sandboxes get sa sandbox-agent-sa -o yaml | grep role-arn` and confirm the role's trust policy permits the cluster's OIDC provider for `system:serviceaccount:agent-sandboxes:sandbox-agent-sa`.
- **conformance.sh fails with "No egress allowlist found"** — install `infra/agent-egress-chained/` or `infra/agent-egress-native/` before running the agent. The reference agent assumes at least one is present.
- **Sandbox pod crashes on startup with "exec format error"** — the gVisor runsc binary didn't install correctly on the node. Check the EC2 console logs for the NodeClaim, look for download failures from `storage.googleapis.com/gvisor`.
