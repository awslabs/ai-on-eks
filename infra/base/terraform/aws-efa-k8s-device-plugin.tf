locals {
  aws_efa_k8s_device_plugin_values = templatefile("${path.module}/helm-values/aws-efa-k8s-device-plugin-values.yaml", {
  })
}

#---------------------------------------------------------------
# AWS EFA K8s DEVICE PLUGIN
#---------------------------------------------------------------
resource "kubectl_manifest" "aws_efa_k8s_device_plugin" {
  count = var.enable_aws_efa_k8s_device_plugin && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/aws-efa-k8s-device-plugin.yaml", {
    version          = var.aws_efa_k8s_device_plugin_version,
    user_values_yaml = indent(8, local.aws_efa_k8s_device_plugin_values)
  })

  depends_on = [
    helm_release.argocd
  ]
}
