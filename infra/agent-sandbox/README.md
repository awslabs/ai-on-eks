# Agent Sandbox on EKS — Blueprint

Deploys an EKS cluster that supports running AI agents inside kernel-isolated sandboxes with FQDN-aware egress filtering. Built on [kubernetes-sigs/agent-sandbox](https://github.com/kubernetes-sigs/agent-sandbox) + Cilium (chaining mode) + Karpenter.

This blueprint is the Phase 1 POC shape — scope is standard EKS + gVisor runtime tier + chained Cilium egress. See `.daedalus-work/pending/agent-sandboxes-on-eks/SPECIFICATION.md` for the full project scope and `.daedalus-work/pending/agent-sandboxes-on-eks/IMPLEMENTATION_PLAN.md` for the execution sequence.

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

- AWS credentials with permissions for VPC, EKS, IAM, EC2, and (for the demo agent) `bedrock:InvokeModel` on the target Claude model.
- `terraform >=1.9`, `kubectl >=1.30`, `helm >=3.18`, `aws` CLI v2.
- An IAM role with `bedrock:InvokeModel` + trust policy allowing EKS Pod Identity — documented in `walkthrough.sh setup`.

## Usage

Install everything:

```bash
cd infra/agent-sandbox
./install.sh            # runs cluster + cilium + sandbox + manifests in order
```

Or run one phase at a time (useful during demo prep):

```bash
./install.sh cluster    # base EKS cluster only (20-30 min)
./install.sh cilium     # + Cilium chaining + Hubble (3-5 min)
./install.sh sandbox    # + kubernetes-sigs/agent-sandbox controller (1-2 min)
./install.sh manifests  # + RuntimeClass, NodePool, SandboxTemplates, policies (~30s)
./install.sh kro        # + KRO + AgentSandbox RGD (optional, stretch)
```

Each phase is idempotent. Re-run `./install.sh manifests` to re-apply after edits.

Deploy the reference agent:

```bash
cd ../../blueprints/agents/sandbox-demo
CLUSTER_NAME=agent-sandbox \
BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
    ./walkthrough.sh setup
./walkthrough.sh run
```

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
    ├── demo-agent.yaml                         # Direct Sandbox resource for the showcase agent
    ├── rgd-agent-sandbox.yaml                  # KRO ResourceGraphDefinition — composes Sandbox + SA + policy
    ├── agent-sandbox-demo-instance.yaml        # AgentSandbox instance via the KRO RGD (stretch demo)
    └── kro-install.sh                          # Installs KRO on the cluster
```

Reference agent lives under `blueprints/agents/sandbox-demo/` — matches the existing ai-on-eks convention where infra lives in `infra/` and workload patterns live in `blueprints/`.

## Design decisions

See `.daedalus-work/pending/agent-sandboxes-on-eks/SPECIFICATION.md` for the full rationale. Short version:

- **Standard EKS, not Auto Mode** — the customer-voiced problem is what standard-EKS teams do today. Native `ApplicationNetworkPolicy` is Auto-Mode-only until AWS extends it; chained Cilium bridges the gap.
- **Raw manifest install for agent-sandbox** — upstream doesn't ship a Helm chart; we match their install pattern rather than wrapping.
- **AL2023 over Bottlerocket for gVisor nodes** — the gVisor containerd shim install scripts upstream target AL2023 systemd. Bottlerocket would require a custom variant build which is out of POC scope.
- **Hubble as primary observability surface** — spec §6.4 rules out Tetragon for this scope. Cilium + Hubble is one dependency covering both enforcement and audit.
- **kro composition pattern is in POC but stretch-only** — adapts `composing-sandbox-nw-policies/rgd.yaml` from upstream. Customers see the AgentSandbox single-CR story without being forced into kro adoption.

## Known limitations (POC today)

1. **gVisor shim install via Karpenter user-data adds 60-90s to first-pod-on-node latency.** Pre-warm the gVisor NodePool before a demo by applying a throwaway pod with `runtimeClassName: gvisor` ~5 minutes before showtime.
2. **Kata + Firecracker tier is not in the POC.** Phase 1 Production will add it (blocked upstream on EKS MNG nested-virt support for now — SPEC §6.6).
3. **Native egress (`ApplicationNetworkPolicy`) is not demonstrated live.** The blueprint is standard EKS + chained Cilium; native egress lives in a separate `agents/egress-native/` blueprint (Phase 1 Production scope).
4. **No snapshot / warm-pool demo.** Phase 2 per SPEC.

## Troubleshooting

- **gVisor pod stays Pending** — Karpenter may be waiting for a suitable instance. Check `kubectl get nodepools` + `kubectl describe nodeclaim`. If the pool is sized out, bump `limits.cpu/memory` in `karpenter-nodepool-gvisor.yaml`.
- **Bedrock call fails with `AccessDeniedException`** — check the Pod Identity association via `aws eks list-pod-identity-associations --cluster-name agent-sandbox`, and confirm the role has `bedrock:InvokeModel` for the exact model ARN.
- **Hubble shows NO flows for the sandbox** — confirm Cilium chaining took effect: `kubectl -n kube-system exec ds/cilium -- cilium status | grep Enforcement`. Expected: "Policy Enforcement: Default".
- **Sandbox pod crashes on startup with "exec format error"** — the gVisor runsc binary didn't install correctly on the node. Check the EC2 console logs for the NodeClaim, look for download failures from `storage.googleapis.com/gvisor`.
