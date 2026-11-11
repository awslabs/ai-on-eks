locals {
  nvidia_gpu_operator_values = yamldecode(templatefile("${path.module}/helm-values/nvidia-gpu-operator.yaml", {
    enable_cluster_policy                 = !(var.enable_nvidia_dra_driver && var.enable_nvidia_gpu_operator && !var.enable_eks_auto_mode)
    enable_gpu_cluster                    = var.enable_nvidia_dra_driver && var.enable_nvidia_gpu_operator && !var.enable_eks_auto_mode
    enable_device_plugin                  = var.enable_nvidia_device_plugin && var.enable_nvidia_gpu_operator && !var.enable_eks_auto_mode
    enable_mig_manager                    = var.enable_nvidia_gpu_operator_mig_manager && !var.enable_eks_auto_mode
    enable_dcgm_exporter                  = var.enable_nvidia_gpu_operator_dcgm_exporter
    enable_eks_auto_mode                  = var.enable_eks_auto_mode
    dcgm_exporter_service_monitor_enabled = local.nvidia_dcgm_exporter_service_monitor
    dcgm_exporter_metrics                 = local.nvidia_dcgm_exporter_metrics
  }))
}

resource "kubectl_manifest" "nvidia_gpu_operator_yaml" {
  count = var.enable_nvidia_gpu_operator ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-gpu-operator.yaml", {
    version          = var.nvidia_gpu_operator_version
    user_values_yaml = indent(8, yamlencode(local.nvidia_gpu_operator_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
