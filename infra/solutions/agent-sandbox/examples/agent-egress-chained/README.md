# Agent Egress (Chained) — Example

Installs [Cilium](https://cilium.io/) in aws-cni chaining mode plus [Hubble](https://github.com/cilium/hubble) for FQDN-based egress enforcement on top of a Standard EKS cluster. Pairs with the parent [agent-sandbox solution](../../).

## When to use this example

Running on Standard EKS, where native `ApplicationNetworkPolicy` is not yet available. The chained Cilium path provides the same FQDN-based egress capability today, with a clean migration to native policies when AWS extends `ApplicationNetworkPolicy` to Standard EKS (announced December 2025, timing TBD).

For EKS Auto Mode, use the sibling [agent-egress-native](../agent-egress-native/) example instead.

## Positioning

Cilium is one of several service meshes that can chain on top of VPC CNI for FQDN filtering and advanced enforcement — Istio, Linkerd, and others support similar patterns. Cilium is used here for convenience (one dependency covers both enforcement and observability via Hubble, CNCF-graduated status, smaller operational surface than mesh alternatives), not out of architectural necessity.

## Prerequisites

- The parent [agent-sandbox solution](../../) installed (provides the cluster, `agent-sandboxes` namespace, and agent-sandbox controller).
- `helm >=3.18`, `kubectl >=1.30`, `aws` CLI v2.
- `kubectl` configured for the target cluster.

## Usage

Full install (Cilium + policies):

```bash
cd infra/solutions/agent-sandbox/examples/agent-egress-chained
./install.sh
```

Phased (useful when iterating on policies):

```bash
./install.sh cilium     # Cilium chaining + Hubble (3-5 min)
./install.sh policies   # Admin + app-tier CNPs (~10s)
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

Each allowlist selects pods by the `allowlist: <name>` label. A pod without any `allowlist` label falls under the default sandbox CNP (`sandbox-llm-allowlist`), which covers the reference agent's needs.

Four shipped templates under `manifests/allowlists/`:

| Template | Destinations |
|----------|--------------|
| `aws-services.yaml` | STS, Bedrock, S3, DynamoDB |
| `llm-apis.yaml` | Bedrock, Anthropic, OpenAI |
| `dev-tools.yaml` | GitHub, GitLab, Docker Hub, ECR, Hugging Face |
| `package-registries.yaml` | PyPI, npm, Maven Central, Go proxy, crates.io, RubyGems |

Each file has a comment header describing what's included and pointing at the equivalent `ApplicationNetworkPolicy` under [`../agent-egress-native/manifests/allowlists/`](../agent-egress-native/manifests/allowlists/) for the eventual chained-to-native migration.

## Directory layout

```
agent-egress-chained/
├── README.md                                    # This file
├── install.sh                                   # Phased installer (cilium | policies | install | uninstall)
└── manifests/
    ├── ciliumclusterwidenetworkpolicy-admin.yaml   # Admin tier: deny IMDS (cluster-wide CNP)
    ├── ciliumnetworkpolicy-sandbox-llm.yaml         # App tier: default sandbox allowlist
    └── allowlists/
        ├── aws-services.yaml
        ├── llm-apis.yaml
        ├── dev-tools.yaml
        └── package-registries.yaml
```

## Hubble observability

After installation, Hubble UI is available via port-forward:

```bash
kubectl port-forward -n kube-system svc/hubble-ui 12000:80
# Open http://localhost:12000 and filter to namespace=agent-sandboxes
```

**Known limitation — FQDN blocks do not appear as DROPPED flows.** Cilium enforces FQDN policy via DNS proxy (returns empty answer for denied domains); the pod never attempts a TCP connection, so no L3/L4 flow is generated for Hubble to visualize. Use `cilium observe` from the Cilium agent pod to see DNS proxy verdicts directly:

```bash
CILIUM_POD=$(kubectl -n kube-system get pods -l k8s-app=cilium -o jsonpath='{.items[0].metadata.name}')
kubectl -n kube-system exec $CILIUM_POD -c cilium-agent -- cilium monitor --type l7 2>&1 | grep "DNS proxy"
```

L3/L4 blocks (raw-IP egress not covered by FQDN allowlist) DO appear as DROPPED flows in the default Hubble Service Map. The [reference agent](../../../../../blueprints/agents/agent-sandbox/) has a Step 5 that exercises this path explicitly to produce a visible red DROP flow.

## Migration path to native ApplicationNetworkPolicy

When AWS extends `ApplicationNetworkPolicy` to Standard EKS:

1. Apply the same allowlists from [`../agent-egress-native/manifests/allowlists/`](../agent-egress-native/manifests/allowlists/) — they're ApplicationNetworkPolicy equivalents of the CNPs shipped here. Pod labels (`allowlist: <name>`) are identical.
2. Delete the CNPs from this example.
3. Uninstall Cilium via `./install.sh uninstall` (optional — Cilium can stay for Hubble observability even when enforcement moves to native).
