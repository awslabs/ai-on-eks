# Agent Egress (Chained) — Blueprint

Adds FQDN-aware egress enforcement and flow observability to an EKS cluster by chaining [Cilium](https://cilium.io/) on top of the VPC CNI. Ships a default admin-tier policy (deny IMDS + link-local), a sandbox-tier allowlist (Bedrock + STS + PyPI), and a small library of additional allowlist templates.

**When to use this blueprint**: running on Standard EKS, where native `ApplicationNetworkPolicy` is not yet available. The chained Cilium path provides the same FQDN-based egress capability today, with a clean migration to native policies when Amazon extends `ApplicationNetworkPolicy` support to Standard EKS.

**When to use the native alternative**: running on EKS Auto Mode, where `ApplicationNetworkPolicy` is available. See `infra/agent-egress-native/`.

Cilium is one of several service meshes that can chain on top of VPC CNI for extended enforcement features — Istio, Linkerd, and others support similar FQDN-filtering patterns. Cilium is used here for convenience (one dependency covers both enforcement and observability via Hubble, CNCF graduated status, and a smaller operational surface than mesh alternatives), not out of architectural necessity.

## What gets installed

| Layer | Component | Notes |
|---|---|---|
| CNI chain | Cilium 1.19+ in `aws-cni` chaining mode | Attaches eBPF programs on top of VPC CNI's veth pair |
| Observability | Hubble relay + UI + metrics | Service map, flow logs, DNS proxy verdicts |
| Admin-tier policy | `CiliumClusterwideNetworkPolicy` | Blocks IMDS + ECS task metadata cluster-wide for the `agent-sandboxes` namespace |
| App-tier policy | `CiliumNetworkPolicy` | Default sandbox allowlist: Bedrock + STS + PyPI |
| Allowlist templates | 4 additional CNPs | LLM APIs, package registries, dev tools, AWS services — apply the ones your agents need |

## Prerequisites

- `infra/agent-sandbox/` installed (provides the `agent-sandboxes` namespace + sandbox controller). This blueprint expects to run on top of that base.
- `helm >=3.18`, `kubectl >=1.30`, `aws` CLI v2.

## Usage

```bash
cd infra/agent-egress-chained
./install.sh            # runs cilium + policies
```

Or phase-by-phase:

```bash
./install.sh cilium     # Cilium chaining + Hubble (3-5 min)
./install.sh policies   # Admin-tier CCNP + app-tier CNP (~10s)
```

Open the Hubble UI:

```bash
kubectl port-forward -n kube-system svc/hubble-ui 12000:80
# then open http://localhost:12000
```

Uninstall (leaves the sandbox blueprint intact):

```bash
./install.sh uninstall
```

## Applying additional allowlists

Label the pods that should be covered, then apply the matching template:

```bash
# Example: a pod that needs to reach LLM APIs AND package registries.
kubectl label pod my-agent -n agent-sandboxes \
    allowlist=llm-apis allowlist=package-registries --overwrite

kubectl apply -f manifests/allowlists/llm-apis.yaml
kubectl apply -f manifests/allowlists/package-registries.yaml
```

Each allowlist selects pods by the `allowlist: <name>` label. A pod without any `allowlist` label falls under the default sandbox CNP (`sandbox-llm-allowlist`), which covers the reference agent's needs. To compose multiple allowlists on the same pod, apply multiple label values and apply each matching CNP.

See `manifests/allowlists/` for the four shipped templates (`aws-services.yaml`, `llm-apis.yaml`, `dev-tools.yaml`, `package-registries.yaml`). Each file has a comment header describing what's included and pointing at the equivalent `ApplicationNetworkPolicy` under `infra/agent-egress-native/`.

## Directory layout

```
infra/agent-egress-chained/
├── README.md                                     # This file
├── install.sh                                    # Phased installer (cilium | policies | all | uninstall)
└── manifests/
    ├── ciliumclusterwidenetworkpolicy-admin.yaml # Admin tier: deny IMDS + link-local, allow DNS
    ├── ciliumnetworkpolicy-sandbox-llm.yaml      # App tier: default sandbox allowlist (Bedrock + STS + PyPI)
    └── allowlists/
        ├── aws-services.yaml                     # STS, Bedrock, S3, DynamoDB
        ├── llm-apis.yaml                         # Bedrock, Anthropic, OpenAI
        ├── dev-tools.yaml                        # GitHub, GitLab, Docker Hub, ECR, Hugging Face
        └── package-registries.yaml               # PyPI, npm, Maven, Go, crates.io, RubyGems
```

## Observability surface

Cilium enforces at two distinct layers, and Hubble surfaces them differently:

- **FQDN enforcement** (L7 / DNS proxy) — Cilium intercepts DNS queries. Denied FQDNs get an empty DNS answer; the pod's resolver sees a lookup failure and never attempts a TCP connection. These verdicts are NOT rendered in Hubble UI's default Service Map view (the UI blacklists DNS events). To observe:
  ```bash
  kubectl -n kube-system exec ds/cilium -c cilium-agent -- cilium monitor --type l7 | grep "DNS proxy"
  ```
- **L3/L4 enforcement** — eBPF drops SYN packets at the socket/network layer for denied IP destinations. These render as red DROPPED flows in Hubble's default Service Map.

The reference agent at `blueprints/agents/agent-sandbox/agent.py` exercises both layers (Step 4 = FQDN denial, Step 5 = L3/L4 denial) so you can see the two surfaces side by side.

## Known limitations

1. **Cilium chaining + VPC CNI adds ~1-2 MB of per-node memory overhead** and a per-node bpf program attach on the veth pair. Acceptable for agent workloads; audit before applying cluster-wide on latency-sensitive production pipelines.
2. **Hubble UI blacklists DNS events by default** — the reference agent's Step 4 (FQDN block) does not render in the Service Map. See the agent's README for workarounds (filter tuning + `cilium observe`).
3. **Multi-allowlist composition requires per-pod label stacking** — Cilium evaluates CNPs independently; a pod labeled `allowlist=llm-apis,allowlist=package-registries` gets the union of both. The `endpointSelector` on each template uses `matchLabels: {allowlist: <name>}` rather than set-based selectors so composition is explicit.

## Migrating to native egress

When `ApplicationNetworkPolicy` extends to Standard EKS (AWS-announced timing "in the coming weeks" from December 2025 at the time of this blueprint's publication):

1. Apply the same allowlists from `infra/agent-egress-native/manifests/allowlists/` — they're ApplicationNetworkPolicy equivalents of the CNPs shipped here.
2. Delete the CNPs from this blueprint.
3. Uninstall Cilium via `./install.sh uninstall` (optional — Cilium can stay for Hubble observability even when enforcement moves to native).

The label convention (`allowlist: <name>`) is identical across both blueprints, so pod-level label annotations don't change.
