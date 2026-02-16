name                             = "inference-cluster"
enable_kuberay_operator          = true
enable_ai_ml_observability_stack = true
enable_aibrix_stack              = true
enable_leader_worker_set         = true
solution_description             = "Guidance for Automated Deployment of Inference ready Amazon EKS Clusters"
solution_id                      = "SO9615"
availability_zones_count         = 4
enable_soci_snapshotter          = true
# AWS region and API version seetings below
# region                           = "us-west-2"
# eks_cluster_version              = "1.34"
# to enable AutoMode, uncomment the setting below
# enable_eks_auto_mode             = true

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

enable_cluster_addons = {
  coredns                         = true
  kube-proxy                      = true
  vpc-cni                         = true
  eks-pod-identity-agent          = true
  aws-ebs-csi-driver              = false
  metrics-server                  = false
  eks-node-monitoring-agent       = false
  amazon-cloudwatch-observability = false
}
