locals {
  istio_values = yamldecode(templatefile("${path.module}/helm-values/istiod.yaml", {
    enable_inference_extension = var.enable_gateway_api_inference_crds
  }))
}

resource "kubectl_manifest" "istio_base_yaml" {
  count = var.enable_istio ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/istio-base.yaml", {
    version = var.istio_version
  })

  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "istiod_yaml" {
  count = var.enable_istio ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/istiod.yaml", {
    version          = var.istio_version
    user_values_yaml = indent(8, yamlencode(local.istio_values))
  })

  depends_on = [
    helm_release.argocd,
    kubectl_manifest.istio_base_yaml
  ]
}
