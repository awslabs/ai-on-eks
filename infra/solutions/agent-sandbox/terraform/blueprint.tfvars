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

# Standard EKS (not Auto Mode) is the default compute mode. gVisor
# shim installation is handled by a Karpenter NodePool (under
# manifests/) that installs containerd-shim-runsc-v1 via AL2023
# user-data — Auto Mode does not expose equivalent node-level hooks.
# To use Auto Mode instead, flip this flag and skip the
# karpenter-nodepool-gvisor manifest; note that gVisor tier is not
# available on Auto Mode.
enable_eks_auto_mode = false
