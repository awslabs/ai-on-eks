locals {
  nvidia_dcgm_exporter_metrics = templatefile("${path.module}/monitoring/dcgm-exporter-metrics.csv", {})

  nvidia_dcgm_exporter_values = yamldecode(templatefile("${path.module}/helm-values/nvidia-dcgm-exporter.yaml", {
    dcgm_exporter_service_monitor_enabled = var.nvidia_dcgm_exporter_service_monitor
    dcgm_exporter_metrics                 = local.nvidia_dcgm_exporter_metrics
  }))
}

resource "kubectl_manifest" "nvidia_dcgm_exporter_yaml" {
  count = var.enable_nvidia_dcgm_exporter && (!var.enable_nvidia_gpu_operator || !var.enable_nvidia_gpu_operator_dcgm_exporter) ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-dcgm-exporter.yaml", {
    version          = var.nvidia_dcgm_exporter_version
    user_values_yaml = indent(8, yamlencode(local.nvidia_dcgm_exporter_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
