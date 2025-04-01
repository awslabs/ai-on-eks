#---------------------------------------------------------------
# Helm Release: NIM Operator for Kubernetes Inference Management
#
# This block installs the NVIDIA Inference Microservice (NIM) Operator
# using Helm. The NIM Operator manages the lifecycle of NIM CRDs like
# NIMService and NIMCache, enabling scalable and efficient model serving.
#
# Note:
# - Helm does not automatically upgrade CRDs.
# - Follow NVIDIA's upgrade guide if updating the operator version:
#   https://docs.nvidia.com/nim-operator/latest/upgrade.html
#---------------------------------------------------------------
resource "helm_release" "nim_operator" {
  name             = "nim-operator"
  namespace        = "nim-operator"
  create_namespace = true
  repository       = "https://helm.ngc.nvidia.com/nvidia"
  chart            = "k8s-nim-operator"
  version          = "v1.0.1"
  timeout          = 360
  wait             = true
}

#---------------------------------------------------------------
# AWS EFS Module: Shared Persistent Storage for Model Caching
#
# This module provisions an Amazon EFS (Elastic File System)
# with mount targets across availability zones. It is used as
# ReadWriteMany shared storage for NIMCache, allowing cached
# models to persist across pod restarts and be reused by multiple
# NIMService pods for faster startup and reduced cold boot latency.
#---------------------------------------------------------------
module "efs" {
  source  = "terraform-aws-modules/efs/aws"
  version = "~> 1.6"

  creation_token = local.name
  name           = local.name

  # Mount targets / security group
  mount_targets = {
    for k, v in zipmap(local.azs, slice(module.vpc.private_subnets, length(module.vpc.private_subnets) - 2, length(module.vpc.private_subnets))) : k => { subnet_id = v }
  }
  security_group_description = "${local.name} EFS security group"
  security_group_vpc_id      = module.vpc.vpc_id
  security_group_rules = {
    vpc = {
      # relying on the defaults provided for EFS/NFS (2049/TCP + ingress)
      description = "NFS ingress from VPC private subnets"
      cidr_blocks = module.vpc.private_subnets_cidr_blocks
    }
  }

  tags = local.tags
}

#---------------------------------------------------------------
# Kubernetes StorageClass for Dynamic EFS Provisioning
#
# This StorageClass enables dynamic provisioning of EFS volumes
# using the AWS EFS CSI driver. It is used by NIMCache CRD to create
# a PersistentVolumeClaim (PVC) with ReadWriteMany access mode,
# essential for sharing cached models between GPU pods in NIMService.
#---------------------------------------------------------------
resource "kubernetes_storage_class_v1" "efs" {
  metadata {
    name = "efs-sc-dynamic"
  }

  storage_provisioner = "efs.csi.aws.com"
  parameters = {
    provisioningMode = "efs-ap" # Dynamic provisioning
    fileSystemId     = module.efs.id
    directoryPerms   = "777"
  }

  mount_options = [
    "iam"
  ]

  depends_on = [
    module.eks_blueprints_addons.aws_efs_csi_driver
  ]
}
