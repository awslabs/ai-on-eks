locals {
  dranet_driver_values = yamldecode(templatefile("${path.module}/helm-values/dranet-driver.yaml", {
    image = var.dranet_driver_image
  }))
}

resource "kubectl_manifest" "dranet_driver_yaml" {
  count = var.enable_dranet_driver && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/dranet-driver.yaml", {
    version          = var.dranet_driver_version
    user_values_yaml = indent(8, yamlencode(local.dranet_driver_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
