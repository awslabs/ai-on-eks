# Agent Egress (Native) — Example

Applies native EKS `ClusterNetworkPolicy` + `ApplicationNetworkPolicy` for FQDN-based egress enforcement on top of an EKS Auto Mode cluster. Pairs with the parent [agent-sandbox solution](../../).

## When to use this example

Running on EKS Auto Mode, want native network policy enforcement without a third-party CNI dependency.

For Standard EKS, use the sibling [agent-egress-chained](../agent-egress-chained/) example instead. When AWS extends `ApplicationNetworkPolicy` to Standard EKS, customers can migrate from chained to native by replacing the CNP manifests with the ANP equivalents shipped here. Pod-level labels (`allowlist: <name>`) are identical across both examples.

## Positioning

Uses the DNS-based FQDN filtering available in EKS Auto Mode (per [AWS docs](https://docs.aws.amazon.com/eks/latest/userguide/auto-net-pol.html)) via the VPC CNI's Network Policy Controller. No Cilium dependency, no service mesh required. Enforcement happens at the VPC CNI's eBPF hooks.

## Prerequisites

- The parent [agent-sandbox solution](../../) installed with `enable_eks_auto_mode = true` in its `terraform/blueprint.tfvars` (provides an Auto Mode cluster + `agent-sandboxes` namespace + agent-sandbox controller).
- `kubectl >=1.30`, `aws` CLI v2.
- `kubectl` configured for the Auto Mode cluster.

### Enforcement requirements

EKS Auto Mode ships the `ApplicationNetworkPolicy` / `ClusterNetworkPolicy` CRDs but **disables the Network Policy Controller by default**. `./install.sh` enables the controller via the `amazon-vpc-cni` ConfigMap in `kube-system` before applying the policies. Without that ConfigMap, policies are accepted silently but nothing is enforced (per [Use Network Policies with EKS Auto Mode](https://docs.aws.amazon.com/eks/latest/userguide/auto-net-pol.html)). If you already enforce the controller cluster-wide, the install is idempotent.

### FQDN wildcard limitation

Native `ApplicationNetworkPolicy` accepts `*` **only as the leftmost label** (e.g., `*.amazonaws.com`). Patterns like `bedrock-runtime.*.amazonaws.com` are rejected at admission. The default allowlist enumerates the most-common AWS regions (us-east-1 + us-west-2) explicitly; consumers in other regions should add matching entries. The chained example's `CiliumNetworkPolicy` variant uses `matchPattern` which supports embedded wildcards, so its templates are more compact.

## Usage

Install (enables Network Policy Controller + applies policies):

```bash
cd infra/solutions/agent-sandbox/examples/agent-egress-native
./install.sh
```

Uninstall:

```bash
./install.sh uninstall
```

## Applying additional allowlists

Label the pods that should be covered, then apply the matching template:

```bash
kubectl label pod my-agent -n agent-sandboxes \
    allowlist=llm-apis --overwrite

kubectl apply -f manifests/allowlists/llm-apis.yaml
```

Each allowlist selects pods by the `allowlist: <name>` label. A pod without any `allowlist` label falls under the default `sandbox-llm-allowlist` ANP, which covers the reference agent's needs.

Four shipped templates under `manifests/allowlists/`:

| Template | Destinations |
|----------|--------------|
| `aws-services.yaml` | STS, Bedrock, S3, DynamoDB (us-east-1 + us-west-2) |
| `llm-apis.yaml` | Bedrock (us-east-1 + us-west-2), Anthropic, OpenAI |
| `dev-tools.yaml` | GitHub, GitLab, Docker Hub, ECR (us-east-1 + us-west-2), Hugging Face |
| `package-registries.yaml` | PyPI, npm, Maven Central, Go proxy, crates.io, RubyGems |

Each file has a comment header describing what's included and pointing at the equivalent `CiliumNetworkPolicy` under [`../agent-egress-chained/manifests/allowlists/`](../agent-egress-chained/manifests/allowlists/).

## Directory layout

```
agent-egress-native/
├── README.md                                      # This file
├── install.sh                                     # Installer (install | uninstall)
└── manifests/
    ├── network-policy-controller-enable.yaml      # ConfigMap enabling the Auto Mode NP Controller
    ├── clusternetworkpolicy-admin.yaml            # Admin tier: deny IMDS (cluster-scoped CNP)
    ├── applicationnetworkpolicy-sandbox-llm.yaml  # App tier: default sandbox allowlist
    ├── test-pod.yaml                              # Minimal test pod for validating enforcement
    └── allowlists/
        ├── aws-services.yaml
        ├── llm-apis.yaml
        ├── dev-tools.yaml
        └── package-registries.yaml
```

## Validating enforcement

A minimal test pod ships at `manifests/test-pod.yaml`:

```bash
kubectl apply -f manifests/test-pod.yaml
kubectl -n agent-sandboxes wait --for=condition=Ready pod/egress-test --timeout=120s

# Allowed FQDN (from the default sandbox-llm-allowlist ANP):
kubectl -n agent-sandboxes exec egress-test -- curl -sS -o /dev/null -w '%{http_code}\n' --max-time 5 https://pypi.org

# Blocked FQDN (not in the allowlist — expect DNS failure):
kubectl -n agent-sandboxes exec egress-test -- curl -sS -o /dev/null -w '%{http_code}\n' --max-time 5 https://blocked-example.example.com

# Blocked raw IP (not in the allowlist — expect connection timeout):
kubectl -n agent-sandboxes exec egress-test -- curl -sS -o /dev/null -w '%{http_code}\n' --max-time 5 https://8.8.8.8

kubectl delete -f manifests/test-pod.yaml
```

For the full 5-step reference agent run (exercises both enforcement layers end-to-end), see the [reference agent blueprint](../../../../../blueprints/agents/agent-sandbox/).

## Migration path from chained to native (when ANP extends to Standard EKS)

1. Apply the ANP templates from this example's `manifests/allowlists/` to the existing Standard EKS cluster.
2. Verify ANP enforcement is working (check `aws eks describe-cluster` to confirm the required VPC CNI version is installed, and that the Network Policy Controller is enabled).
3. Remove the Cilium CNPs from [`../agent-egress-chained/`](../agent-egress-chained/).
4. Optionally uninstall Cilium (`../agent-egress-chained/install.sh uninstall`) — Cilium can stay for Hubble observability even when enforcement moves to native.

Pod-level allowlist labels do not change.
