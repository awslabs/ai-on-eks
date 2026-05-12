# Agent Sandbox on EKS — Blueprint

Deploys an EKS cluster that supports running AI agents inside kernel-isolated sandboxes with FQDN-aware egress filtering. Built on [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) + Cilium (chaining mode) + Karpenter.

Scope: standard EKS + gVisor runtime tier + chained Cilium egress. A reference Python agent lives under `../../blueprints/agents/agent-sandbox/` — see that directory's README for how to exercise the blueprint end-to-end.

## What gets installed

| Layer | Component | Source |
|---|---|---|
| Cluster | EKS (standard, not Auto Mode) v1.34 | ai-on-eks base module |
| Workers | Karpenter 1.11+ | ai-on-eks base module |
| CNI | AWS VPC CNI v1.21.1+ in chaining mode + Cilium | Blueprint overlay |
| Observability | Hubble UI + relay | Blueprint overlay |
| Sandbox runtime | kubernetes-sigs/agent-sandbox v0.4.3 | Blueprint overlay |
| Runtime tiers | `standard` (runc) + `gvisor` | Blueprint overlay |
| Policies | CiliumClusterwideNetworkPolicy (admin tier) + CiliumNetworkPolicy (app tier) | Blueprint overlay |
| Composition (optional) | kro AgentSandbox RGD | Blueprint overlay |

## Prerequisites

- AWS credentials with permissions for VPC, EKS, IAM, EC2, and (for the reference agent) `bedrock:InvokeModel` on the target Claude model.
- `terraform >=1.9`, `kubectl >=1.30`, `helm >=3.18`, `aws` CLI v2.
- An IAM role with `bedrock:InvokeModel` + IRSA trust policy allowing the cluster's OIDC provider — templates at `manifests/iam-bedrock-trust-policy.template.json` and `manifests/iam-bedrock-permissions.template.json`.

## Usage

Install everything:

```bash
cd infra/agent-sandbox
./install.sh            # runs cluster + cilium + sandbox + manifests in order
```

Or run one phase at a time:

```bash
./install.sh cluster    # base EKS cluster only (20-30 min)
./install.sh cilium     # + Cilium chaining + Hubble (3-5 min)
./install.sh sandbox    # + kubernetes-sigs/agent-sandbox controller (1-2 min)
./install.sh manifests  # + RuntimeClass, NodePool, SandboxTemplates, policies (~30s)
./install.sh kro        # + KRO + AgentSandbox RGD (optional)
```

Each phase is idempotent. Re-run `./install.sh manifests` to re-apply after edits.

Deploy and run the reference agent:

```bash
cd ../../blueprints/agents/agent-sandbox
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    ./conformance.sh
```

See `../../blueprints/agents/agent-sandbox/README.md` for interactive-run instructions and troubleshooting.

Open the Hubble UI in a second terminal to watch egress decisions live:

```bash
kubectl port-forward -n kube-system svc/hubble-ui 12000:80
# then open http://localhost:12000
```

Destroy:

```bash
cd infra/agent-sandbox/terraform/_LOCAL
./cleanup.sh
```

## Directory layout

```
infra/agent-sandbox/
├── README.md                # This file
├── install.sh               # Phased installer (cluster | cilium | sandbox | manifests | kro | all)
├── terraform/
│   └── blueprint.tfvars     # Feature flags for the base ai-on-eks module
└── manifests/
    ├── namespace.yaml                          # agent-sandboxes namespace with egress-tier label
    ├── runtimeclass-gvisor.yaml                # gVisor RuntimeClass pointing at containerd's runsc handler
    ├── karpenter-nodepool-gvisor.yaml          # Dedicated NodePool with AL2023 + gVisor shim user-data
    ├── sandbox-template-standard.yaml          # Tier 1 SandboxTemplate (runc)
    ├── sandbox-template-gvisor.yaml            # Tier 2 SandboxTemplate (gVisor)
    ├── ciliumclusterwidenetworkpolicy-admin.yaml  # Admin tier: deny IMDS + link-local cluster-wide, allow DNS
    ├── ciliumnetworkpolicy-sandbox-llm.yaml    # App tier: FQDN allowlist (Bedrock + STS + PyPI)
    ├── sandbox-agent.yaml                      # Direct Sandbox resource for the reference agent
    ├── rgd-agent-sandbox.yaml                  # KRO ResourceGraphDefinition — composes Sandbox + SA
    ├── agent-sandbox-instance.yaml             # AgentSandbox instance via the KRO RGD (optional)
    └── kro-install.sh                          # Installs KRO on the cluster
```

Reference agent lives under `blueprints/agents/agent-sandbox/` — matches the existing ai-on-eks convention where infra lives in `infra/` and workload patterns live in `blueprints/`.

## Design decisions

- **Standard EKS, not Auto Mode** — targets the standard-EKS customer reality. Native `ApplicationNetworkPolicy` is Auto-Mode-only today; chained Cilium bridges the gap.
- **Raw manifest install for agent-sandbox** — upstream doesn't ship a Helm chart; this blueprint matches their install pattern rather than wrapping.
- **AL2023 over Bottlerocket for gVisor nodes** — the gVisor containerd shim install scripts upstream target AL2023 systemd. Bottlerocket would require a custom variant build.
- **Hubble as primary observability surface** — Cilium + Hubble is one dependency covering both enforcement and audit.
- **IRSA rather than EKS Pod Identity for gVisor-tier workloads** — Pod Identity's credential endpoint (169.254.170.23) is served on the host network namespace. gVisor's Sentry doesn't forward that link-local route through the sandbox's virtual network stack, so Pod Identity isn't reachable from inside a gVisor sandbox. IRSA (`AWS_WEB_IDENTITY_TOKEN_FILE` + STS) routes through the regular network path, which the Cilium FQDN policy permits via `sts.*.amazonaws.com`.
- **kro composition is optional** — adapts `composing-sandbox-nw-policies/rgd.yaml` from upstream. Customers can use the blueprint without adopting kro.

## Known limitations

1. **gVisor shim install via Karpenter user-data adds 60-90s to first-pod-on-node latency.** Pre-warm the gVisor NodePool by applying a throwaway pod with `runtimeClassName: gvisor` ~5 minutes before you need the first real workload.
2. **Kata + Firecracker tier is not included.** Follow-on work; blocked upstream on EKS MNG nested-virt support.
3. **Native egress (`ApplicationNetworkPolicy`) is not included here.** This blueprint is standard EKS + chained Cilium.

## Troubleshooting

- **gVisor pod stays Pending** — Karpenter may be waiting for a suitable instance. Check `kubectl get nodepools` + `kubectl describe nodeclaim`. If the pool is sized out, bump `limits.cpu/memory` in `karpenter-nodepool-gvisor.yaml`.
- **Bedrock call fails with `AccessDeniedException`** — the IRSA annotation is missing or the IAM role lacks `bedrock:InvokeModel`. Check `kubectl -n agent-sandboxes get sa sandbox-agent-sa -o yaml | grep role-arn` and confirm the role's trust policy permits the cluster's OIDC provider for `system:serviceaccount:agent-sandboxes:sandbox-agent-sa`.
- **Hubble shows no drops for blocked FQDN lookups** — expected. Cilium FQDN enforcement is a DNS-proxy filter (empty answer), not an L3/L4 drop. The reference agent's Step 5 (raw TCP to a non-allowlisted IP) produces a DROPPED flow in Hubble's default view; Step 4's FQDN block is observable via `cilium observe` / DNS proxy logs. See the reference agent's README for details.
- **Hubble shows NO flows at all for the sandbox** — confirm Cilium chaining took effect: `kubectl -n kube-system exec ds/cilium -- cilium status | grep Enforcement`. Expected: `Policy Enforcement: Default`.
- **Sandbox pod crashes on startup with "exec format error"** — the gVisor runsc binary didn't install correctly on the node. Check the EC2 console logs for the NodeClaim, look for download failures from `storage.googleapis.com/gvisor`.
