name                = "agent-sandbox"
region              = "us-east-1"
eks_cluster_version = "1.34"

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
# which pulls v1.21.1+ for DNS-based ClusterNetworkPolicy support. The
# demo also relies on metrics-server for resource visibility during the
# live run.
enable_cluster_addons = {
  coredns                         = true
  kube-proxy                      = true
  vpc-cni                         = true
  eks-pod-identity-agent          = true
  metrics-server                  = true
  eks-node-monitoring-agent       = true
  amazon-cloudwatch-observability = true
}

# AWS Load Balancer Controller — needed for any ingress we expose during
# the demo (Hubble UI, sample agent webhook). Already the base default
# but explicit here for clarity.
enable_aws_load_balancer_controller = true

# Observability — keep the Kube Prometheus stack off for the demo
# (Hubble's built-in UI is enough, and Prometheus adds ~90s to the
# install). Re-enable for long-lived deployments where flow metrics
# matter.
enable_kube_prometheus_stack = false
enable_amazon_prometheus     = false

# EKS Auto Mode add-ons we don't use in this blueprint.
enable_jupyterhub       = false
enable_kuberay_operator = false
enable_mlflow_tracking  = false
enable_argo_workflows   = false
enable_argo_events      = false
