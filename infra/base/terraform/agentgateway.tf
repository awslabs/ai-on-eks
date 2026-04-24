locals {
  agentgateway_values = yamldecode(templatefile("${path.module}/helm-values/agentgateway.yaml", {
    enable_inference_extension = var.enable_gateway_api_inference_crds
  }))
}

resource "kubectl_manifest" "agentgateway_crds_yaml" {
  count = var.enable_agentgateway ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/agentgateway-crds.yaml", {
    version = var.agentgateway_version
  })

  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "agentgateway_yaml" {
  count = var.enable_agentgateway ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/agentgateway.yaml", {
    version          = var.agentgateway_version
    user_values_yaml = indent(8, yamlencode(local.agentgateway_values))
  })

  depends_on = [
    helm_release.argocd,
    kubectl_manifest.agentgateway_crds_yaml
  ]
}
