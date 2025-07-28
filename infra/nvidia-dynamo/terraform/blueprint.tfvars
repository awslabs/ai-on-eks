name                = "dynamo-on-eks"
enable_dynamo_stack = true
enable_argocd       = true
# region              = "us-west-2"
eks_cluster_version = "1.33"

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

# -------------------------------------------------------------------------------------
# Dynamo Cloud Infrastructure Configuration
#
# These settings configure the infrastructure components required for Dynamo Cloud:
# - EFS for shared persistent storage (model caching, shared data)
# - Monitoring stack for observability (Prometheus, Grafana)
# - EFA for high-performance networking (GPU/CPU inference workloads)
# - AI/ML observability for specialized inference monitoring
# -------------------------------------------------------------------------------------

# Enable EFS CSI Driver for shared persistent storage
# Required for Dynamo model caching and shared data volumes
enable_aws_efs_csi_driver = true

# Enable monitoring stack for Dynamo observability
# Includes Prometheus for metrics collection and Grafana for visualization
enable_kube_prometheus_stack = true

# Enable AWS EFA (Elastic Fabric Adapter) for high-performance networking
# Provides low-latency, high-bandwidth networking for GPU/CPU instances
enable_aws_efa_k8s_device_plugin = true

# Enable AI/ML observability stack for enhanced monitoring
# Provides specialized monitoring for ML workloads and model performance
enable_ai_ml_observability_stack = true

# -------------------------------------------------------------------------------------
# Optional: Additional ML/AI Infrastructure Components
#
# These components can be enabled based on your specific Dynamo deployment needs:
# -------------------------------------------------------------------------------------

# Enable MLFlow for experiment tracking (optional)
# enable_mlflow_tracking = true

# Enable JupyterHub for interactive development (optional)
# enable_jupyterhub = true

# Enable Argo Workflows for ML pipelines (optional)
# enable_argo_workflows = true

# Enable FSx for Lustre for high-performance file system (optional)
# enable_aws_fsx_csi_driver = true
# deploy_fsx_volume = true

# Enable Ray Serve High Availability with ElastiCache Redis (optional)
# Provides distributed state management for Ray clusters
# enable_rayserve_ha_elastic_cache_redis = true

# -------------------------------------------------------------------------------------
# Dynamo Stack Configuration
# -------------------------------------------------------------------------------------

# Dynamo version to deploy (v0.3.1 with separate dependencies)
dynamo_stack_version = "v0.3.1"

# Hugging Face token for model downloads (replace with your token)
# huggingface_token = "your-huggingface-token-here"

# -------------------------------------------------------------------------------------
# Custom Karpenter Node Pools Configuration
#
# High-priority node pools with BuildKit support for Dynamo workloads
# These have higher weight (100) than base addons (50) for priority scheduling
# -------------------------------------------------------------------------------------

# Enable custom Karpenter node pools with higher weights and BuildKit support
enable_custom_karpenter_nodepools = true

# AMI family selection: AL2023 (default, recommended) or Bottlerocket
# AL2023: Better compatibility, user namespaces enabled by default
# Bottlerocket: Enhanced security, requires custom user namespace configuration
use_bottlerocket = false

# -------------------------------------------------------------------------------------
# Node Configuration (Optional Overrides)
#
# Additional settings to customize the EKS infrastructure
# -------------------------------------------------------------------------------------

# Bottlerocket data disk snapshot for faster node startup
# bottlerocket_data_disk_snapshot_id = "snap-xxxxxxxxx"



# -------------------------------------------------------------------------------------
# ECR Token Refresh Configuration
#
# Automatic refresh of ECR tokens for docker-imagepullsecret and dynamo-regcred
# -------------------------------------------------------------------------------------

# Enable ECR token refresh CronJob
enable_ecr_token_refresh = true

# Schedule for ECR token refresh (every 6 hours)
ecr_token_refresh_schedule = "0 */6 * * *"
