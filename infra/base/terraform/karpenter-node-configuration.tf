locals {
  ec2nodeclass = templatefile("${path.module}/karpenter-resources/templates/${var.ami_family}_ec2nodeclass.tpl",
    {
      node_iam_role                       = var.enable_eks_auto_mode ? module.eks.node_iam_role_name : module.karpenter[0].node_iam_role_name
      cluster_name                        = module.eks.cluster_name
      enable_soci_snapshotter             = var.enable_soci_snapshotter
      soci_snapshotter_use_instance_store = var.soci_snapshotter_use_instance_store
      data_disk_snapshot_id               = var.bottlerocket_data_disk_snapshot_id
      max_user_namespaces                 = var.max_user_namespaces
    }
  )
  ec2nodeclassnames = distinct(flatten(concat(["g5-nvidia", "g6-nvidia", "g6e-nvidia", "inf2-neuron", "trn1-neuron", "m6i-cpu"], var.karpenter_additional_ec2nodeclassnames)))
  ec2nodeclassmanifests = {
    for name in local.ec2nodeclassnames :
    name => templatefile("${path.module}/karpenter-resources/templates/ec2nodeclass.tpl", {
      name         = name,
      ec2nodeclass = local.ec2nodeclass
    })
  }
  karpenter_node_pools = {
    for name in local.ec2nodeclassnames :
    name => templatefile("${path.module}/karpenter-resources/templates/nodepool.tpl", {
      name            = name,
      instance_family = split("-", name)[0]
      ami_family      = var.ami_family
      taints          = contains(split("-", name), "nvidia") ? "nvidia.com/gpu" : contains(split("-", name), "neuron") ? "aws.amazon.com/neuron" : ""
    })
  }
}

resource "kubectl_manifest" "ec2nodeclass" {
  for_each = !var.enable_eks_auto_mode ? local.ec2nodeclassmanifests : {}

  yaml_body = each.value
  wait      = true
  depends_on = [
    module.karpenter,
    helm_release.karpenter,
    aws_ec2_tag.cluster_primary_security_group
  ]
}

resource "kubectl_manifest" "nodepool" {
  for_each = !var.enable_eks_auto_mode ? local.karpenter_node_pools : {}

  yaml_body = each.value
  wait      = true

  depends_on = [
    module.karpenter,
    helm_release.karpenter,
    kubectl_manifest.ec2nodeclass
  ]
}
