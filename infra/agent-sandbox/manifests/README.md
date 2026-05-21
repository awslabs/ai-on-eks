# Manifests

Platform-layer Kubernetes resources for the agent-sandbox infrastructure. The parent `terraform/` provisions the cluster + ArgoCD addons; the manifests here run on top of that cluster to register runtime tiers, provision compute capacity, and stage the IRSA scaffolding for sandbox workloads.

These are the primitives required for **any** SandboxClaim to land on the cluster. Workload-specific manifests (the reference SandboxClaim, KRO composition example, agent ConfigMap) live in [`blueprints/agent-sandbox/`](../../../blueprints/agent-sandbox/).

## What's here

| Path | Purpose |
|---|---|
| `namespace.yaml` | The `agent-sandboxes` namespace. |
| `runtimeclass-gvisor.yaml` | RuntimeClass + scheduling block for the gVisor tier (Standard EKS only). |
| `karpenter-nodepool-gvisor.yaml` | Karpenter NodePool + EC2NodeClass that supplies gVisor-capable nodes. AL2023 user-data installs `containerd-shim-runsc-v1`. |
| `sandbox-template-standard.yaml` | Basic SandboxTemplate for the standard (runc) tier. Hardened Pod spec with no workload-specific assumptions; default workload is `nginx:alpine` (the K8s shell-demo image). Mode-agnostic — works on both Standard EKS and Auto Mode. |
| `sandbox-template-gvisor.yaml` | Basic SandboxTemplate for the gVisor tier. Same shape as the standard template plus `runtimeClassName: gvisor` and the gVisor NodePool toleration. Standard EKS only (Auto Mode doesn't expose hooks for the runsc shim). |
| `iam/bedrock-trust-policy.template.json` | IRSA trust policy template usable by sandbox workloads that need Bedrock access. Substitute `<ACCOUNT_ID>`, `<REGION>`, `<OIDC_PROVIDER_ID>` (the blueprint's egress example handles this automatically via `install.sh irsa`). |
| `iam/bedrock-permissions.template.json` | Bedrock invoke permissions template, paired with the trust policy above. |

## Adding a new runtime tier

Each tier adds three files, parallel to the gVisor set:

- `runtimeclass-<tier>.yaml` — RuntimeClass + scheduling block
- `karpenter-nodepool-<tier>.yaml` — NodePool + EC2NodeClass with the tier's runtime shim install in user-data
- `sandbox-template-<tier>.yaml` — SandboxTemplate using `runtimeClassName: <tier>` and the matching toleration

A `SandboxClaim` (in any blueprint or your own workload) targets the new tier by setting `sandboxTemplateRef.name` accordingly. No changes elsewhere in this directory.

## Layering on top

The basic SandboxTemplates ship a hardened Pod spec with `nginx:alpine` as the default workload. Anything beyond that — workload-specific image, IRSA-bound ServiceAccount, ConfigMap mounts, env vars — lives in a blueprint that introduces those assumptions.

Two blueprints ship in this repo today:

- [`blueprints/agent-sandbox/basic/`](../../../blueprints/agent-sandbox/basic/) — smallest viable Sandbox deployment, claims one of the basic SandboxTemplates above directly.
- [`blueprints/agent-sandbox/`](../../../blueprints/agent-sandbox/) — reference agent with FQDN egress enforcement, KRO composition, and end-to-end conformance. Ships its own agent-shaped templates ([`sandbox-template-agent-{standard,gvisor}.yaml`](../../../blueprints/agent-sandbox/manifests/)) that mirror the basic shape with the agent's Python + Bedrock + ConfigMap additions.

The pattern generalizes: every distinct workload-shape gets its own SandboxTemplate, every claim picks the template that matches the workload it's wrapping. This keeps the platform primitives (basic templates) clean and extensible, and keeps workload-specific assumptions (image, env, volumes, IRSA) in the blueprint that introduces them. You can equally write your own SandboxClaims that target the basic templates without using either blueprint.
