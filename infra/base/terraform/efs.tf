#---------------------------------------------------------------
# AWS EFS Module: Shared Persistent Storage for Model Caching
#
# This module provisions an Amazon EFS (Elastic File System)
# with mount targets across availability zones.
#---------------------------------------------------------------
module "efs" {
  count   = var.enable_aws_efs_csi_driver ? 1 : 0
  source  = "terraform-aws-modules/efs/aws"
  version = "~> 2.2"

  creation_token = local.name
  name           = local.name

  throughput_mode = "elastic"
  # Mount targets / security group
  mount_targets = {
    for k, v in zipmap(local.azs, slice(module.vpc.private_subnets, length(module.vpc.private_subnets) - local.azs_count, length(module.vpc.private_subnets))) : k => { subnet_id = v }
    if alltrue([for az in local.local_azs : k != az])
  }

  security_group_name        = "${local.name}-efs"
  security_group_description = "${local.name} EFS security group"
  security_group_vpc_id      = module.vpc.vpc_id
  security_group_ingress_rules = {
    for cidr in module.vpc.private_subnets_cidr_blocks : cidr => {
      # relying on the defaults provided for EFS/NFS (2049/TCP + ingress)
      description = "NFS ingress from VPC private subnets"
      cidr_ipv4   = cidr
    }
  }

  tags = local.tags
}

#---------------------------------------------------------------
# Kubernetes StorageClass for Dynamic EFS Provisioning
#
# This StorageClass enables dynamic provisioning of EFS volumes
# using the AWS EFS CSI driver. It is used by NIMCache CRD to create
# a PersistentVolumeClaim (PVC) with ReadWriteMany access mode.
#---------------------------------------------------------------
resource "kubernetes_storage_class_v1" "efs" {
  count = var.enable_aws_efs_csi_driver ? 1 : 0
  metadata {
    name = "efs-sc-dynamic"
  }

  storage_provisioner = "efs.csi.aws.com"
  parameters = {
    provisioningMode = "efs-ap" # Dynamic provisioning
    fileSystemId     = module.efs[0].id
    directoryPerms   = "777"
  }

  mount_options = [
    "iam"
  ]

  depends_on = [
    aws_eks_addon.aws_efs_csi_driver,
    module.efs
  ]
}
