module "karpenter" {
  count   = !var.enable_eks_auto_mode ? 1 : 0
  source  = "terraform-aws-modules/eks/aws//modules/karpenter"
  version = "~> 21.6"

  cluster_name = module.eks.cluster_name
  namespace    = "kube-system"

  iam_role_name = "KarpenterController-${local.name}"

  # Name needs to match role name passed to the EC2NodeClass
  node_iam_role_name              = "karpenterNode-${local.name}"
  create_pod_identity_association = true

  # Managed policy exceeds 6144 chars for longer cluster names; inline allows 10240
  enable_inline_policy = true

  # Used to attach additional IAM policies to the Karpenter node IAM role
  node_iam_role_additional_policies = {
    AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
  }
  tags = local.tags
}

resource "helm_release" "karpenter_crd" {
  count            = !var.enable_eks_auto_mode ? 1 : 0
  name             = "karpenter-crd"
  namespace        = "kube-system"
  create_namespace = true
  repository       = "oci://public.ecr.aws/karpenter"
  chart            = "karpenter-crd"
  version          = var.karpenter_version
  wait             = true
  take_ownership   = true

  depends_on = [module.karpenter]
}

resource "helm_release" "karpenter" {
  count            = !var.enable_eks_auto_mode ? 1 : 0
  name             = "karpenter"
  namespace        = "kube-system"
  create_namespace = true
  repository       = "oci://public.ecr.aws/karpenter"
  chart            = "karpenter"
  version          = var.karpenter_version
  wait             = true
  skip_crds        = true

  values = [
    <<-EOT
    nodeSelector:
      NodeGroupType: 'core'
    settings:
      clusterName: ${module.eks.cluster_name}
      clusterEndpoint: ${module.eks.cluster_endpoint}
      interruptionQueue: ${module.karpenter[0].queue_name}
      featureGates:
        nodeOverlay: true
        spotToSpotConsolidation: true
        staticCapacity: true
        capacityBuffer: true
    tolerations:
      - key: CriticalAddonsOnly
        operator: Exists
      - key: karpenter.sh/controller
        operator: Exists
        effect: NoSchedule
    webhook:
      enabled: false
    EOT
  ]

  depends_on = [helm_release.karpenter_crd]
}
