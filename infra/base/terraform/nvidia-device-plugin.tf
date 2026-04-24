locals {
  nvidia_device_plugin_values = yamldecode(templatefile("${path.module}/helm-values/nvidia-device-plugin.yaml", {}))
}

resource "kubectl_manifest" "nvidia_device_plugin_yaml" {
  count = !var.enable_nvidia_gpu_operator && var.enable_nvidia_device_plugin && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-device-plugin.yaml", {
    version          = var.nvidia_device_plugin_version
    user_values_yaml = indent(8, yamlencode(local.nvidia_device_plugin_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
