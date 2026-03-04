################################################################################
# Cluster
################################################################################

name   = "genai-workshop"
region = "us-east-2" # update to your preferred AWS region # update to your preferred AWS region

################################################################################
# EKS
################################################################################

enable_eks_auto_mode = true

################################################################################
# Observability
################################################################################

enable_kube_prometheus_stack    = true
enable_grafana_operator         = true
enable_amazon_prometheus        = true
enable_nvidia_dcgm_exporter     = false
kube_prometheus_stack_namespace = "monitoring"
grafana_service_port            = 3000
grafana_admin_password          = "notforproductionuse"

################################################################################
# Model Storage (S3)
################################################################################

enable_s3_models_storage     = true
s3_models_bucket_create      = false
s3_models_additional_buckets = ["genai-models-*"]
s3_models_sync_sa            = "model-storage-sa"
