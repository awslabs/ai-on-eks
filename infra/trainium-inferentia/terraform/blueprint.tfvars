# Must stay <= 17 chars: the Karpenter controller IAM role name_prefix is
# "KarpenterController-${name}-" and the AWS provider caps name_prefix at 38.
name                      = "trainium-inf"
enable_aws_efs_csi_driver = true
enable_kuberay_operator   = true
enable_volcano            = true
enable_torchx_etcd        = true
enable_mpi_operator       = true
enable_aws_fsx_csi_driver = true
deploy_fsx_volume         = true

# Advertises vpc.amazonaws.com/efa, which the RayTrain-Llama2 workers request.
enable_aws_efa_k8s_device_plugin = true
# region                    = "us-west-2"
# eks_cluster_version       = "1.34"

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
