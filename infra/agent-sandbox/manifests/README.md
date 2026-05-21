# Manifests

Platform-layer Kubernetes resources for the agent-sandbox infrastructure. The parent `terraform/` provisions the cluster + ArgoCD addons; the manifests here run on top of that cluster to register runtime tiers, provision compute capacity, and stage the IRSA scaffolding for sandbox workloads.

These are the primitives required for **any** SandboxClaim to land on the cluster. Workload-specific manifests (the reference SandboxClaim, KRO composition example, agent ConfigMap) live in [`blueprints/agent-sandbox/`](../../../blueprints/agent-sandbox/).

## What's here

| Path | Purpose |
|---|---|
| `namespace.yaml` | The `agent-sandboxes` namespace. |
| `runtimeclass-gvisor.yaml` | RuntimeClass + scheduling block for the gVisor tier (Standard EKS only). |
| `karpenter-nodepool-gvisor.yaml` | Karpenter NodePool + EC2NodeClass that supplies gVisor-capable nodes. AL2023 user-data installs `containerd-shim-runsc-v1`. |
| `sandbox-template-standard.yaml` | SandboxTemplate for the standard (runc) tier. Mode-agnostic — works on both Standard EKS and Auto Mode. |
| `sandbox-template-gvisor.yaml` | SandboxTemplate for the gVisor tier. Standard EKS only (Auto Mode doesn't expose hooks for the runsc shim). |
| `iam/bedrock-trust-policy.template.json` | IRSA trust policy template usable by sandbox workloads that need Bedrock access. Substitute `<ACCOUNT_ID>`, `<REGION>`, `<OIDC_PROVIDER_ID>` (the blueprint's egress example handles this automatically via `install.sh irsa`). |
| `iam/bedrock-permissions.template.json` | Bedrock invoke permissions template, paired with the trust policy above. |

## Adding a new runtime tier

Each tier adds three files, parallel to the gVisor set:

- `runtimeclass-<tier>.yaml` — RuntimeClass + scheduling block
- `karpenter-nodepool-<tier>.yaml` — NodePool + EC2NodeClass with the tier's runtime shim install in user-data
- `sandbox-template-<tier>.yaml` — SandboxTemplate using `runtimeClassName: <tier>` and the matching toleration

A `SandboxClaim` (in any blueprint or your own workload) targets the new tier by setting `sandboxTemplateRef.name` accordingly. No changes elsewhere in this directory.

## Layering on top

Anything beyond the platform primitives — the reference agent, KRO composition, egress enforcement, end-to-end conformance — lives in the blueprint. See [`../../../blueprints/agent-sandbox/`](../../../blueprints/agent-sandbox/) for the canonical example. The blueprint is one consumer of this infra; you can equally point your own SandboxClaims at the templates installed here.
