# Agent Egress — Example

Mode-aware FQDN egress enforcement for the agent-sandbox blueprint. Auto-detects whether the cluster is Standard EKS or EKS Auto Mode and applies the right enforcement layer accordingly. Pairs with the parent [agent-sandbox blueprint](../).

## How it works

| Cluster mode | Enforcement | Observability |
|---|---|---|
| Standard EKS | [Cilium](https://cilium.io/) `CiliumClusterwideNetworkPolicy` + `CiliumNetworkPolicy` (chained on AWS VPC CNI) | [Hubble](https://github.com/cilium/hubble) UI for flow visibility |
| EKS Auto Mode | Native `ClusterNetworkPolicy` + `ApplicationNetworkPolicy` (DNS-based, enforced by VPC CNI Network Policy Controller) | VPC Flow Logs + CloudWatch |

`./install.sh` queries the AWS API at apply time to determine compute mode, then chooses the right manifests and integration steps. Pod-level allowlist labels (`allowlist: <name>`) are identical across both backends — agent workloads are portable between modes without relabeling.

## Prerequisites

- The [agent-sandbox infrastructure](../../../infra/agent-sandbox/) installed.
- For **Standard EKS**: `enable_cilium = true` in the infra's `terraform/blueprint.tfvars` (the base infra ArgoCD-deploys Cilium chaining mode + Hubble — this example does *not* install Cilium itself).
- For **EKS Auto Mode**: `enable_eks_auto_mode = true` and `enable_cilium = false` in `blueprint.tfvars` (no third-party CNI required).
- `kubectl >=1.30`, `aws` CLI v2, `jq`.
- `kubectl` configured for the target cluster.

### Standard EKS positioning

Cilium is one of several CNIs that can chain on top of VPC CNI for FQDN filtering — Calico and others support similar patterns. Cilium is used in this blueprint for convenience (one dependency covers both enforcement and observability via Hubble, CNCF-graduated status, smaller operational surface than alternatives), not out of architectural necessity.

### Auto Mode requirements

EKS Auto Mode ships the `ApplicationNetworkPolicy` / `ClusterNetworkPolicy` CRDs but **disables the Network Policy Controller by default**. `./install.sh` enables the controller via the `amazon-vpc-cni` ConfigMap in `kube-system` before applying the policies. Without that ConfigMap, policies are accepted silently but nothing is enforced (per [Use Network Policies with EKS Auto Mode](https://docs.aws.amazon.com/eks/latest/userguide/auto-net-pol.html)). If you already enforce the controller cluster-wide, the install is idempotent.

### Auto Mode FQDN wildcard limitation

Native `ApplicationNetworkPolicy` accepts `*` **only as the leftmost label** (e.g., `*.amazonaws.com`). Patterns like `bedrock-runtime.*.amazonaws.com` are rejected at admission. The default allowlist enumerates the most-common AWS regions (us-east-1 + us-west-2) explicitly; consumers in other regions should add matching entries. Cilium's `matchPattern` supports embedded wildcards, so the Cilium templates are more compact.

## Usage

Full install (mode detection + policies + Bedrock IRSA role):

```bash
cd blueprints/agent-sandbox/egress
./install.sh
```

Phased:

```bash
./install.sh policies   # Apply egress policies for the detected mode
./install.sh irsa       # Bedrock IRSA role only (idempotent — refreshes trust policy on cluster recreation)
```

Uninstall (removes policies + IRSA role; leaves Cilium / Network Policy Controller in place since they may be shared with other workloads):

```bash
./install.sh uninstall
```

The `irsa` phase provisions a Bedrock IAM role named `<cluster-name>-bedrock-irsa` (default: `agent-sandbox-bedrock-irsa`) with the trust policy scoped to the live cluster's OIDC provider. It's idempotent and safe to re-run after a cluster recreation — the trust policy gets refreshed with the new OIDC ID. The role ARN is echoed at the end for use with `conformance.sh`.

## Applying additional allowlists

Label the pods that should be covered, then apply the matching template from the right backend's directory:

```bash
# Pick the directory matching your enforcement backend:
ls manifests/allowlists/cilium/   # Standard EKS
ls manifests/allowlists/anp/      # Auto Mode

# Label and apply:
kubectl label pod my-agent -n agent-sandboxes \
    allowlist=llm-apis --overwrite

# Standard EKS:
kubectl apply -f manifests/allowlists/cilium/llm-apis.yaml

# Auto Mode:
kubectl apply -f manifests/allowlists/anp/llm-apis.yaml
```

Each allowlist selects pods by the `allowlist: <name>` label. A pod without any `allowlist` label falls under the default sandbox CNP/ANP (`sandbox-llm-allowlist`), which covers the reference agent's needs.

Four shipped allowlist templates per backend:

| Template | Destinations |
|----------|--------------|
| `aws-services.yaml` | STS, Bedrock, S3, DynamoDB |
| `llm-apis.yaml` | Bedrock, Anthropic, OpenAI |
| `dev-tools.yaml` | GitHub, GitLab, Docker Hub, ECR, Hugging Face |
| `package-registries.yaml` | PyPI, npm, Maven Central, Go proxy, crates.io, RubyGems |

## Directory layout

```
egress/
├── README.md                                          # This file
├── install.sh                                         # Mode-aware installer
└── manifests/
    ├── cilium/                                        # Standard EKS
    │   ├── ciliumclusterwidenetworkpolicy-admin.yaml  # Admin tier: deny IMDS
    │   └── ciliumnetworkpolicy-sandbox-llm.yaml       # App tier: default sandbox allowlist
    ├── anp/                                           # EKS Auto Mode
    │   ├── network-policy-controller-enable.yaml      # ConfigMap enabling the NP Controller
    │   ├── clusternetworkpolicy-admin.yaml            # Admin tier: deny IMDS
    │   └── applicationnetworkpolicy-sandbox-llm.yaml  # App tier: default sandbox allowlist
    └── allowlists/
        ├── cilium/                                    # CNP versions of the four allowlists
        │   ├── aws-services.yaml
        │   ├── llm-apis.yaml
        │   ├── dev-tools.yaml
        │   └── package-registries.yaml
        └── anp/                                       # ANP versions of the same four allowlists
            ├── aws-services.yaml
            ├── llm-apis.yaml
            ├── dev-tools.yaml
            └── package-registries.yaml
```

## Hubble observability (Standard EKS only)

After installation on Standard EKS, Hubble UI is available via port-forward:

```bash
kubectl port-forward -n kube-system svc/hubble-ui 12000:80
# Open http://localhost:12000 and filter to namespace=agent-sandboxes
```

**Known limitation — FQDN blocks do not appear as DROPPED flows.** Cilium enforces FQDN policy via DNS proxy (returns empty answer for denied domains); the pod never attempts a TCP connection, so no L3/L4 flow is generated for Hubble to visualize. Use `cilium observe` from the Cilium agent pod to see DNS proxy verdicts directly:

```bash
CILIUM_POD=$(kubectl -n kube-system get pods -l k8s-app=cilium -o jsonpath='{.items[0].metadata.name}')
kubectl -n kube-system exec $CILIUM_POD -c cilium-agent -- cilium monitor --type l7 2>&1 | grep "DNS proxy"
```

L3/L4 blocks (raw-IP egress not covered by FQDN allowlist) DO appear as DROPPED flows in the default Hubble Service Map. The [reference agent](../) has a Step 5 that exercises this path explicitly to produce a visible red DROP flow.

## Validating enforcement

Run the reference agent's conformance test — it auto-detects compute mode, claims the right agent-shaped SandboxTemplate (`sandbox-agent-runc` on Auto Mode, `sandbox-agent-gvisor` on Standard EKS), and asserts all 5 PASS / BLOCKED outcomes including Step 4 (FQDN block) and Step 5 (raw IP block).

```bash
cd ..
BEDROCK_ROLE_ARN=$(aws iam get-role --role-name agent-sandbox-bedrock-irsa --query 'Role.Arn' --output text) \
    ./conformance.sh
```

For ad-hoc validation against the policy without the full reference agent, exec into any pod in the `agent-sandboxes` namespace with the `egress-tier: sandbox` label and run `curl` / `socket.connect` tests.

## Migrating between modes

If you migrate an existing cluster from Standard EKS to Auto Mode (or vice versa), the same example covers both — flip `enable_eks_auto_mode` and `enable_cilium` in the infra's `blueprint.tfvars`, re-run the infra's `./install.sh`, and re-run this example's `./install.sh`. The mode detection picks up the new compute mode and applies the right enforcement layer.

Pod-level allowlist labels do not change — agent workloads are portable across both backends.
