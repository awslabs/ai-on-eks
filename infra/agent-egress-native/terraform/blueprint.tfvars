name                = "agent-egress-native"
region              = "us-east-1"
eks_cluster_version = "1.34"

# Region explicitly declared above. Change to your preferred AWS
# region before running `install.sh`. Can also be overridden via the
# `TF_VAR_region` environment variable without editing this file.
#
# (The ai-on-eks base module defaults to us-west-2 when `region` is
# omitted; this blueprint pins us-east-1 because
# ApplicationNetworkPolicy availability and the agent-sandbox
# reference agent's Bedrock IAM role default to that region.)

# EKS Auto Mode is the supported deployment target for
# ApplicationNetworkPolicy. The DNS-based FQDN filtering this
# blueprint relies on is available ONLY on EKS Auto Mode-launched
# EC2 instances today (per AWS docs). For Standard EKS, use
# infra/agent-egress-chained/ instead.
enable_eks_auto_mode = true

# Auto Mode handles most cluster addons natively. Keep the list
# short; Auto Mode's own compute layer provides VPC CNI, kube-proxy,
# CoreDNS, and metrics-server.
enable_cluster_addons = {
  coredns                         = true
  kube-proxy                      = true
  vpc-cni                         = true
  eks-pod-identity-agent          = true
  metrics-server                  = true
  eks-node-monitoring-agent       = true
  amazon-cloudwatch-observability = true
}

enable_aws_load_balancer_controller = true

# Observability add-ons — keep off by default. Auto Mode's built-in
# observability covers the basics.
enable_kube_prometheus_stack = false
enable_amazon_prometheus     = false
