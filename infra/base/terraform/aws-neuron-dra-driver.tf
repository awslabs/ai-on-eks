locals {
  aws_neuron_dra_driver_values = yamldecode(templatefile("${path.module}/helm-values/aws-neuron-dra-driver.yaml", {}))
}

# AWS Neuron DRA (Dynamic Resource Allocation) driver.
# Delivered as its own ArgoCD application (separate from the Neuron device
# plugin) because the DRA driver and the device plugin must not run on the same
# node. The driver is pinned to nodes labelled neuron.aws/driver=dra (the
# neuron-dra-static NodePool); the device plugin stays on neuron.aws/driver=
# device-plugin nodes. Requires Kubernetes >= 1.34.
resource "kubectl_manifest" "aws_neuron_dra_driver" {
  count = var.enable_aws_neuron_dra_driver && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/aws-neuron-dra-driver.yaml", {
    version          = var.aws_neuron_dra_driver_version
    user_values_yaml = indent(8, yamlencode(local.aws_neuron_dra_driver_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
