# ECR Repository Outputs (only when Dynamo stack is enabled)
output "dynamo_operator_repository_url" {
  description = "The URL of the Dynamo operator ECR repository"
  value       = var.enable_dynamo_stack ? aws_ecr_repository.dynamo_operator[0].repository_url : null
}

output "dynamo_api_store_repository_url" {
  description = "The URL of the Dynamo API store ECR repository"
  value       = var.enable_dynamo_stack ? aws_ecr_repository.dynamo_api_store[0].repository_url : null
}

output "dynamo_pipelines_repository_url" {
  description = "The URL of the Dynamo pipelines ECR repository"
  value       = var.enable_dynamo_stack ? aws_ecr_repository.dynamo_pipelines[0].repository_url : null
}

output "dynamo_base_repository_url" {
  description = "The URL of the Dynamo base ECR repository"
  value       = var.enable_dynamo_stack ? aws_ecr_repository.dynamo_base[0].repository_url : null
}

#---------------------------------------------------------------
# Outputs for Custom Karpenter Node Pools
#---------------------------------------------------------------

output "custom_karpenter_c7i_nodeclass_name" {
  description = "Name of the custom C7i CPU node class"
  value       = var.enable_custom_karpenter_nodepools ? kubectl_manifest.dynamo_c7i_cpu_nodeclass[0].name : null
}

output "custom_karpenter_c7i_nodepool_name" {
  description = "Name of the custom C7i CPU node pool"
  value       = var.enable_custom_karpenter_nodepools ? kubectl_manifest.dynamo_c7i_cpu_nodepool[0].name : null
}

output "custom_karpenter_g6_nodeclass_name" {
  description = "Name of the custom G6 GPU node class"
  value       = var.enable_custom_karpenter_nodepools ? kubectl_manifest.dynamo_g6_gpu_nodeclass[0].name : null
}

output "custom_karpenter_g6_nodepool_name" {
  description = "Name of the custom G6 GPU node pool"
  value       = var.enable_custom_karpenter_nodepools ? kubectl_manifest.dynamo_g6_gpu_nodepool[0].name : null
}

output "custom_karpenter_node_labels" {
  description = "Labels applied to custom Karpenter nodes"
  value = {
    c7i_cpu = {
      "dynamo.ai/node-type"           = "c7i-cpu"
      "dynamo.ai/buildkit-compatible" = "true"
      "type"                          = "karpenter"
      "instanceType"                  = "dynamo-c7i-cpu"
    }
    g6_gpu = {
      "dynamo.ai/node-type"           = "g6-gpu"
      "dynamo.ai/buildkit-compatible" = "true"
      "type"                          = "karpenter"
      "instanceType"                  = "dynamo-g6-gpu"
      "accelerator"                   = "nvidia"
      "gpuType"                       = "l4"
    }
  }
}
