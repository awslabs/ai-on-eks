# Manifests

Workload-layer Kubernetes resources for the agent-sandbox solution. The parent solution's `terraform/` provisions the cluster + ArgoCD addons; the manifests here run on top of that cluster to register the runtime tier, provision compute capacity for it, and stage the reference agent.

## What's here

| Path | Purpose |
|---|---|
| `namespace.yaml` | The `agent-sandboxes` namespace. |
| `runtimeclass-gvisor.yaml` | RuntimeClass + scheduling block for the gVisor tier (Standard EKS only). |
| `karpenter-nodepool-gvisor.yaml` | Karpenter NodePool + EC2NodeClass that supplies gVisor-capable nodes. AL2023 user-data installs `containerd-shim-runsc-v1`. |
| `mng-sample/launch-template.yaml` | Sample EKS Managed Node Group launch template — alternative to Karpenter for the gVisor tier. |
| `sandbox-template-standard.yaml` | SandboxTemplate for the standard (runc) tier. Mode-agnostic — works on both Standard EKS and Auto Mode. |
| `sandbox-template-gvisor.yaml` | SandboxTemplate for the gVisor tier. Standard EKS only (Auto Mode doesn't expose hooks for the runsc shim). |
| `sandbox-agent.yaml` | Reference SandboxClaim + ServiceAccount + agent-script ConfigMap. The claim's `templateRef.name` is patched at apply time (`sandbox-gvisor` on Standard EKS, `sandbox-standard` on Auto Mode). |
| `kro/rgd.yaml` | KRO ResourceGraphDefinition that exposes a single `AgentSandbox` CRD wrapping the same workload shape. |
| `kro/instance.yaml` | Sample AgentSandbox instance. `__RUNTIME_CLASS__` and `__BEDROCK_ROLE_ARN__` are patched at apply time. |
| `iam/bedrock-trust-policy.template.json` | IRSA trust policy template for the reference agent's Bedrock role. Substitute `<ACCOUNT_ID>`, `<REGION>`, `<OIDC_PROVIDER_ID>` (handled automatically by the egress example's `install.sh irsa` phase). |
| `iam/bedrock-permissions.template.json` | Bedrock invoke permissions template. |

## Adding a new runtime tier

Each tier adds three files, parallel to the gVisor set:

- `runtimeclass-<tier>.yaml` — RuntimeClass + scheduling block
- `karpenter-nodepool-<tier>.yaml` — NodePool + EC2NodeClass with the tier's runtime shim install in user-data
- `sandbox-template-<tier>.yaml` — SandboxTemplate using `runtimeClassName: <tier>` and the matching toleration

A `SandboxClaim` (or `AgentSandbox`) targets the new tier by setting `templateRef.name` (or `runtimeClass`) accordingly. No changes elsewhere in the manifests directory.

## Why two reference paths

`sandbox-agent.yaml` is the SandboxClaim → SandboxTemplate path, native to the SIG-Apps `agent-sandbox` API. The runtime spec lives in the template; the claim is thin per-deployment glue.

`kro/instance.yaml` is the KRO composition path. The `AgentSandbox` CRD wraps the equivalent shape into a single user-facing resource. Useful when exposing a simpler surface to teams that don't need the full Sandbox API.

Both paths produce equivalent running pods. Customers pick whichever shape matches their team.
