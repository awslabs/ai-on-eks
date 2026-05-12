name                = "agent-sandbox"
region              = "us-east-1"
eks_cluster_version = "1.34"

# Region explicitly declared above. Change to your preferred AWS
# region before running `install.sh`. Can also be overridden via the
# `TF_VAR_region` environment variable without editing this file.
#
# (The ai-on-eks base module defaults to us-west-2 when `region` is
# omitted; this blueprint pins us-east-1 because Bedrock model
# availability and the reference agent's IAM trust policy example
# target that region. Blueprint consumers pick their own.)

# Use standard EKS (not Auto Mode) so the runtime class story works
# cleanly. gVisor shim installation needs node-level control which is
# Karpenter's domain, not Auto Mode's.
enable_eks_auto_mode = false

# Karpenter runs on Bottlerocket by default. gVisor nodes use the AL2023
# NodePool variant (see manifests/karpenter-nodepool-gvisor.yaml — a
# post-base manifest) because the gVisor containerd shim integrates
# more predictably with AL2023 than Bottlerocket today.
ami_family = "bottlerocket"

# Core addons — the base already ships VPC CNI with most_recent=true,
# which pulls v1.21.1+ for DNS-based ClusterNetworkPolicy support.
# metrics-server is retained for `kubectl top` visibility during
# development and troubleshooting.
enable_cluster_addons = {
  coredns                         = true
  kube-proxy                      = true
  vpc-cni                         = true
  eks-pod-identity-agent          = true
  metrics-server                  = true
  eks-node-monitoring-agent       = true
  amazon-cloudwatch-observability = true
}

# AWS Load Balancer Controller — needed for any ingress that exposes
# cluster services externally (e.g., Hubble UI or an agent webhook).
# Already the base default but explicit here for clarity.
enable_aws_load_balancer_controller = true

# Observability — keep the Kube Prometheus stack off by default
# (Hubble's built-in UI covers the flow-level observability this
# blueprint needs, and Prometheus adds ~90s to the install). Enable
# for long-lived deployments where flow metrics retention matters.
enable_kube_prometheus_stack = false
enable_amazon_prometheus     = false

# EKS Auto Mode add-ons we don't use in this blueprint.
enable_jupyterhub       = false
enable_kuberay_operator = false
enable_mlflow_tracking  = false
enable_argo_workflows   = false
enable_argo_events      = false
