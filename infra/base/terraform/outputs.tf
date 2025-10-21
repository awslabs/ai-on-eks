output "configure_kubectl" {
  description = "Configure kubectl: make sure you're logged in with the correct AWS profile and run the following command to update your kubeconfig"
  value       = "aws eks --region ${var.region} update-kubeconfig --name ${var.name}"
}

output "grafana_secret_name" {
  description = "The name of the secret containing the Grafana admin password."
  value       = var.enable_kube_prometheus_stack ? aws_secretsmanager_secret.grafana[0].name : null
}

output "fsx_s3_bucket_name" {
  description = "Name of the S3 bucket for FSx"
  value       = var.deploy_fsx_volume ? module.fsx_s3_bucket[0].s3_bucket_id : null
}

# S3 Model Storage Outputs - Only output values when feature is enabled
output "s3_models_buckets_name" {
  description = "Name of the S3 models buckets"
  value       = var.enable_s3_models_storage ? flatten([concat([local.s3_models_bucket_name], var.s3_models_additional_buckets)]) : null
}

output "s3_models_sync_sa" {
  description = "Name of the model sync service account"
  value       = var.enable_s3_models_storage ? var.s3_models_sync_sa : null
}

output "s3_models_inference_sa" {
  description = "Name of the model inference service account"
  value       = var.enable_s3_models_storage ? var.s3_models_inference_sa : null
}

output "s3_models_sync_sa_namespace" {
  description = "Namespace for model sync service account"
  value       = var.enable_s3_models_storage ? var.s3_models_sync_sa_namespace : null
}

output "s3_models_inference_sa_namespace" {
  description = "Namespace for model inference service account"
  value       = var.enable_s3_models_storage ? var.s3_models_inference_sa_namespace : null
}
