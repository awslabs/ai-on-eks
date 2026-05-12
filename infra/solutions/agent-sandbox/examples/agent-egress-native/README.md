# Agent Egress (Native) — Blueprint

Applies native EKS network policies (`ClusterNetworkPolicy` + `ApplicationNetworkPolicy`) for FQDN-aware egress enforcement. Requires **EKS Auto Mode** — the DNS-based FQDN filtering this blueprint relies on is available only on Auto Mode-launched EC2 instances today.

**When to use this blueprint**: running on EKS Auto Mode, want native network policy enforcement without a third-party CNI dependency.

**When to use the chained alternative**: running on Standard EKS, where `ApplicationNetworkPolicy` is not yet available. See `infra/agent-egress-chained/`. When AWS extends ANP to Standard EKS, customers can migrate from chained to native by replacing the CNP manifests with the ANP equivalents shipped here. Pod-level labels (`allowlist: <name>`) are identical across both blueprints.

This blueprint can provision its own EKS Auto Mode cluster (via the standard ai-on-eks base module with `enable_eks_auto_mode = true`) or apply policies to an existing Auto Mode cluster.

## What gets installed

| Layer | Component | Notes |
|---|---|---|
| Cluster (optional) | EKS Auto Mode v1.34 | Via `./install.sh cluster`. Skip this phase if you already have an Auto Mode cluster. |
| Admin-tier policy | `ClusterNetworkPolicy` | Blocks IMDS + ECS task metadata for the `agent-sandboxes` namespace |
| App-tier policy | `ApplicationNetworkPolicy` | Default sandbox allowlist: Bedrock + STS + PyPI |
| Allowlist templates | 4 additional ANPs | LLM APIs, package registries, dev tools, AWS services — apply the ones your agents need |

## Prerequisites

- An existing EKS Auto Mode cluster (or provision one via `./install.sh cluster`).
- `infra/agent-sandbox/` installed against the same cluster (provides the `agent-sandboxes` namespace + sandbox controller).
- `kubectl >=1.30`, `aws` CLI v2.

Note on enforcement: EKS Auto Mode ships the `ApplicationNetworkPolicy` / `ClusterNetworkPolicy` CRDs but disables the Network Policy Controller by default. `./install.sh policies` enables the controller via the `amazon-vpc-cni` ConfigMap in `kube-system` before applying the policies. Without that ConfigMap, policies are accepted silently but nothing is enforced (per [Use Network Policies with EKS Auto Mode](https://docs.aws.amazon.com/eks/latest/userguide/auto-net-pol.html)). If you already enforce the controller cluster-wide, the install is idempotent.

Note on FQDN wildcards: native ApplicationNetworkPolicy accepts `*` only as the leftmost label (e.g., `*.amazonaws.com`). Patterns like `bedrock-runtime.*.amazonaws.com` are rejected at admission. The default allowlist therefore enumerates the most-common AWS regions (us-east-1 + us-west-2) explicitly; consumers in other regions should add matching entries. The chained blueprint's `CiliumNetworkPolicy` variant uses `matchPattern` which supports embedded wildcards, so its templates are more compact.

## Usage

Apply policies to an existing Auto Mode cluster:

```bash
cd infra/agent-egress-native
./install.sh policies       # ~10s
```

Provision a new Auto Mode cluster + apply policies:

```bash
cd infra/agent-egress-native
./install.sh cluster        # ~20-30 min
./install.sh policies       # ~10s
```

Uninstall (leaves the cluster intact):

```bash
./install.sh uninstall
```

Destroy the cluster (only if provisioned by this blueprint):

```bash
cd terraform/_LOCAL
./cleanup.sh
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

Each allowlist selects pods by the `allowlist: <name>` label. A pod without any `allowlist` label falls under the default `sandbox-llm-allowlist` ANP, which covers the reference agent's needs.

See `manifests/allowlists/` for the four shipped templates (`aws-services.yaml`, `llm-apis.yaml`, `dev-tools.yaml`, `package-registries.yaml`). Each file has a comment header describing what's included and pointing at the equivalent `CiliumNetworkPolicy` under `infra/agent-egress-chained/`.

## Directory layout

```
infra/agent-egress-native/
├── README.md                                     # This file
├── install.sh                                    # Phased installer (cluster | policies | all | uninstall)
├── terraform/
│   └── blueprint.tfvars                          # EKS Auto Mode enabled
└── manifests/
    ├── network-policy-controller-enable.yaml    # ConfigMap enabling the Auto Mode NP Controller
    ├── clusternetworkpolicy-admin.yaml           # Admin tier: deny IMDS (Admin tier CNP)
    ├── applicationnetworkpolicy-sandbox-llm.yaml # App tier: default sandbox allowlist (Bedrock + STS + PyPI)
    ├── test-pod.yaml                             # Minimal test pod for validating enforcement
    └── allowlists/
        ├── aws-services.yaml                     # STS, Bedrock, S3, DynamoDB
        ├── llm-apis.yaml                         # Bedrock, Anthropic, OpenAI
        ├── dev-tools.yaml                        # GitHub, GitLab, Docker Hub, ECR, Hugging Face
        └── package-registries.yaml               # PyPI, npm, Maven, Go, crates.io, RubyGems
```

## Observability

Native EKS network policies integrate with CloudWatch Logs (configurable via NodeClass `networkPolicyEventLogs: Enabled`). For richer flow observability, install Hubble separately or chain Cilium in observability-only mode alongside the native enforcement path (Cilium ships with its own Hubble dashboard).

See the AWS documentation for ANP operational notes: https://docs.aws.amazon.com/eks/latest/userguide/auto-net-pol.html

## Known limitations

1. **Requires EKS Auto Mode** for DNS-based FQDN enforcement. Applying these manifests to Standard EKS creates the CRDs (once `ApplicationNetworkPolicy` support extends to Standard) but DNS-based rules won't enforce until AWS ships full Standard EKS support.
2. **Policy evaluation order matters** — Admin tier policies (this blueprint's `ClusterNetworkPolicy`) are evaluated before namespace-scoped policies. An Admin Deny cannot be overridden by namespace-level ANPs. See the AWS documentation on policy evaluation order.
3. **FQDN enforcement is DNS-proxy-based** — just like the chained variant. Denied FQDN lookups produce an empty DNS answer; the pod's resolver sees a hostname failure. No L3/L4 DROPPED flow is generated because the pod never attempts a TCP connection to the denied FQDN.

## Migrating from chained to native

When `ApplicationNetworkPolicy` extends to Standard EKS:

1. Apply the ANP templates from this blueprint's `manifests/allowlists/` to the existing Standard EKS cluster.
2. Verify ANP enforcement is working (check `aws eks describe-cluster` to confirm the required VPC CNI version is installed).
3. Remove the Cilium CNPs from `infra/agent-egress-chained/manifests/`.
4. Optionally uninstall Cilium (`./install.sh uninstall` from `infra/agent-egress-chained/`) — Cilium can stay for Hubble observability even when enforcement moves to native.

Pod-level allowlist labels do not change.
