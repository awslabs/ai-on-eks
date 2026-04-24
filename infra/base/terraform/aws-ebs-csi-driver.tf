resource "aws_eks_addon" "aws_ebs_csi_driver" {
  count                       = !var.enable_eks_auto_mode ? 1 : 0
  cluster_name                = module.eks.cluster_name
  addon_name                  = "aws-ebs-csi-driver"
  resolve_conflicts_on_update = "PRESERVE"
  depends_on                  = [module.aws_ebs_csi_pod_identity]
}

module "aws_ebs_csi_pod_identity" {
  count   = !var.enable_eks_auto_mode ? 1 : 0
  source  = "terraform-aws-modules/eks-pod-identity/aws"
  version = "~> 2.2"
  name    = "aws-ebs-csi"

  attach_aws_ebs_csi_policy = true
  aws_ebs_csi_kms_arns      = ["arn:aws:kms:*:*:key/${module.eks.cluster_name}-ebs-csi-kms-key"]

  associations = {
    ebs_csi = {
      cluster_name    = module.eks.cluster_name
      namespace       = "kube-system"
      service_account = "ebs-csi-controller-sa"
    }
  }

  tags = local.tags
}
