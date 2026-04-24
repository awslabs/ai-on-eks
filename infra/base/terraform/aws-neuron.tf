resource "kubectl_manifest" "neuron_monitor" {
  count      = var.enable_aws_neuron_device_plugin ? 1 : 0
  yaml_body  = file("${path.module}/monitoring/neuron-monitor-daemonset.yaml")
  depends_on = [kubectl_manifest.aws_neuron_device_plugin]
}

resource "kubectl_manifest" "aws_neuron_device_plugin" {
  count = var.enable_aws_neuron_device_plugin ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/aws-neuron-device-plugin.yaml", {
    version = var.aws_neuron_device_plugin_version
  })

  depends_on = [
    helm_release.argocd,
  ]
}
