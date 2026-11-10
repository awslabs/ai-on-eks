locals {
  aws_neuron_device_plugin_values = templatefile("${path.module}/helm-values/aws-neuron-device-plugin-values.yaml", {})
}

resource "kubectl_manifest" "neuron_monitor" {
  count = (var.enable_aws_neuron_device_plugin || var.enable_eks_auto_mode) && var.enable_aws_neuron_monitor ? 1 : 0
  yaml_body = templatefile("${path.module}/monitoring/neuron-monitor-daemonset.yaml", {
    version = var.aws_neuron_monitor_version
  })
  depends_on = [kubectl_manifest.aws_neuron_device_plugin]
}

resource "kubectl_manifest" "aws_neuron_device_plugin" {
  count = var.enable_aws_neuron_device_plugin && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/aws-neuron-device-plugin.yaml", {
    version          = var.aws_neuron_device_plugin_version,
    user_values_yaml = indent(8, local.aws_neuron_device_plugin_values)
  })

  depends_on = [
    helm_release.argocd,
  ]
}
