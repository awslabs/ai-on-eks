# Grove (gang scheduling operator for multi-node AI workloads)
# Installed from the upstream OCI Helm chart with GitOps overrides
# (crdInstaller.enabled=true, webhookServerSecret.enabled=false) so CRDs are
# installed via init container and the operator manages its own webhook TLS
# Secret instead of the chart overwriting it on every sync.
# See https://github.com/ai-dynamo/grove/blob/main/docs/installation.md
locals {
  grove_values = yamldecode(file("${path.module}/helm-values/grove.yaml"))
}

resource "kubectl_manifest" "grove_yaml" {
  count = var.enable_grove ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/grove.yaml", {
    version          = var.grove_version
    user_values_yaml = indent(8, yamlencode(local.grove_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
