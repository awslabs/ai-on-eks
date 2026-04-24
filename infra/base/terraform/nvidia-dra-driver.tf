locals {
  nvidia_dra_driver_values = yamldecode(templatefile("${path.module}/helm-values/nvidia-dra-driver.yaml", {}))
}

resource "kubectl_manifest" "nvidia_dra_driver_yaml" {
  count = var.enable_nvidia_dra_driver && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-dra-driver.yaml", {
    version          = var.nvidia_dra_driver_version
    user_values_yaml = indent(8, yamlencode(local.nvidia_dra_driver_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
