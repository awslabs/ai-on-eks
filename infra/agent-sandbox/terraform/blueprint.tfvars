name                = "agent-sandbox"
eks_cluster_version = "1.34"

# region              = "us-west-2" # set to the region where your target Bedrock model is available

# Agent Sandbox primitives — ArgoCD-deployed via the base module
# (see infra/base/terraform/argocd_addons.tf). The controller enables
# Sandbox / SandboxTemplate / SandboxClaim CRDs + reconciler; kro
# enables ResourceGraphDefinition-based composition so customers can
# expose a simpler AgentSandbox CR to their teams. Both are optional
# at the base level and opted-in by this solution.
enable_agent_sandbox = true
enable_kro           = true

# Cilium in aws-cni chaining mode — required for chained-egress FQDN
# enforcement on Standard EKS. Auto Mode users should set this to
# false and rely on native ApplicationNetworkPolicy instead (the
# example install.sh auto-detects compute mode and applies the right
# enforcement layer accordingly).
enable_cilium = true

# Standard EKS (not Auto Mode) is the default compute mode. gVisor
# shim installation is handled by a Karpenter NodePool (under
# manifests/) that installs containerd-shim-runsc-v1 via AL2023
# user-data — Auto Mode does not expose equivalent node-level hooks.
# To use Auto Mode instead: flip this flag, set enable_cilium=false,
# and skip the karpenter-nodepool-gvisor manifest. Note that gVisor
# tier is not available on Auto Mode.
enable_eks_auto_mode = false
