name                      = "nvidia-nim-eks"
enable_aws_efs_csi_driver = true
enable_argocd             = true
enable_nvidia_nim_stack   = true
# region                    = "us-west-2"  #set to appropriate region
# eks_cluster_version       = "1.34" #set appropriate eks cluster version

# -------------------------------------------------------------------------------------
# Enable this to NVIDIA K8s DRA Driver with NVIDIA GPU Opeator
#   Check infra/base/terraform/variables.tf for more details
# -------------------------------------------------------------------------------------
enable_nvidia_dra_driver   = true
enable_nvidia_gpu_operator = true

# -------------------------------------------------------------------------------------
# EKS Addons Configuration
#
# These are the EKS Cluster Addons managed by Terraform stack.
# You can enable or disable any addon by setting the value to `true` or `false`.
#
# If you need to add a new addon that isn't listed here:
# 1. Add the addon name to the `enable_cluster_addons` variable in `base/terraform/variables.tf`
# 2. Update the `locals.cluster_addons` logic in `eks.tf` to include any required configuration
#
# -------------------------------------------------------------------------------------

# enable_cluster_addons = {
#   coredns                         = true
#   kube-proxy                      = true
#   vpc-cni                         = true
#   eks-pod-identity-agent          = true
#   aws-ebs-csi-driver              = true
#   metrics-server                  = true
#   eks-node-monitoring-agent       = false
#   amazon-cloudwatch-observability = true
# }
