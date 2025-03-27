#-----------------------------------------------------------------------------------------
# JupyterHub Single User IRSA, maybe that block could be incorporated in add-on registry
#-----------------------------------------------------------------------------------------
resource "kubernetes_namespace" "jupyterhub" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name = "jupyterhub"
  }
}

module "jupyterhub_single_user_irsa" {
  count = var.enable_jupyterhub ? 1 : 0
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"

  role_name = "${module.eks.cluster_name}-jupyterhub-single-user-sa"

  role_policy_arns = {
    policy = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess" # Policy needs to be defined based in what you need to give access to your notebook instances.
  }

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["${kubernetes_namespace.jupyterhub[count.index].metadata[0].name}:jupyterhub-single-user"]
    }
  }
}

resource "kubernetes_service_account_v1" "jupyterhub_single_user_sa" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name        = "${module.eks.cluster_name}-jupyterhub-single-user"
    namespace   = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
    annotations = { "eks.amazonaws.com/role-arn" : module.jupyterhub_single_user_irsa[0].iam_role_arn }
  }

  automount_service_account_token = true
}

resource "kubernetes_secret_v1" "jupyterhub_single_user" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = "${module.eks.cluster_name}-jupyterhub-single-user-secret"
    namespace = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
    annotations = {
      "kubernetes.io/service-account.name"      = kubernetes_service_account_v1.jupyterhub_single_user_sa[count.index].metadata[0].name
      "kubernetes.io/service-account.namespace" = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
    }
  }

  type = "kubernetes.io/service-account-token"
}

#---------------------------------------------------------------
# EFS Filesystem for private volumes per user
# This will be replaced with Dynamic EFS provision using EFS CSI Driver
#---------------------------------------------------------------
resource "aws_efs_file_system" "efs" {
  count = var.enable_jupyterhub ? 1 : 0
  encrypted = true

  tags = local.tags
}

#---------------------------------------------------------------
# module.vpc.private_subnets = [AZ1_10.x, AZ2_10.x, AZ1_100.x, AZ2_100.x]
# We use index 2 and 3 to select the subnet in AZ1 with the 100.x CIDR:
# Create EFS mount targets for the 3rd  subnet
resource "aws_efs_mount_target" "efs_mt_1" {
  count = var.enable_jupyterhub ? 1 : 0
  file_system_id  = aws_efs_file_system.efs[count.index].id
  subnet_id       = module.vpc.private_subnets[2]
  security_groups = [aws_security_group.efs[count.index].id]
}

# Create EFS mount target for the 4th subnet
resource "aws_efs_mount_target" "efs_mt_2" {
  count = var.enable_jupyterhub ? 1 : 0
  file_system_id  = aws_efs_file_system.efs[count.index].id
  subnet_id       = module.vpc.private_subnets[3]
  security_groups = [aws_security_group.efs[count.index].id]
}

resource "aws_security_group" "efs" {
  count = var.enable_jupyterhub ? 1 : 0
  name        = "${local.name}-efs"
  description = "Allow inbound NFS traffic from private subnets of the VPC"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "Allow NFS 2049/tcp"
    cidr_blocks = module.vpc.vpc_secondary_cidr_blocks
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
  }

  tags = local.tags
}

#---------------------------------------
# EFS Configuration
#---------------------------------------
resource "aws_efs_access_point" "efs_persist_ap" {
  count = var.enable_jupyterhub ? 1 : 0
  file_system_id = aws_efs_file_system.efs[count.index].id
  posix_user {
    gid = 0
    uid = 0
    secondary_gids = [100]
  }
  root_directory {
    path = "/home"
    creation_info {
      owner_gid   = 0
      owner_uid   = 0
      permissions = 700
    }
  }
}
resource "aws_efs_access_point" "efs_shared_ap" {
  count = var.enable_jupyterhub ? 1 : 0
  file_system_id = aws_efs_file_system.efs[count.index].id
  posix_user {
    gid = 0
    uid = 0
    secondary_gids = [100]
  }
  root_directory {
    path = "/shared"
    creation_info {
      owner_gid   = 0
      owner_uid   = 0
      permissions = 700
    }
  }
}

module "efs_config" {
  count = var.enable_jupyterhub ? 1 : 0
  source  = "aws-ia/eks-blueprints-addons/aws"
  version = "~> 1.20"

  cluster_name      = module.eks.cluster_name
  cluster_endpoint  = module.eks.cluster_endpoint
  cluster_version   = module.eks.cluster_version
  oidc_provider_arn = module.eks.oidc_provider_arn

  helm_releases = {
    efs = {
      name             = "efs"
      description      = "A Helm chart for storage configurations"
      namespace        = "jupyterhub"
      create_namespace = false
      chart            = "${path.module}/helm-values/efs"
      chart_version    = "0.0.1"
      values = [
        <<-EOT
          pv:
            name: efs-persist
            volumeHandle: ${aws_efs_file_system.efs[count.index].id}::${aws_efs_access_point.efs_persist_ap[count.index].id}
          pvc:
            name: efs-persist
        EOT
      ]
    }
    efs-shared = {
      name             = "efs-shared"
      description      = "A Helm chart for shared storage configurations"
      namespace        = "jupyterhub"
      create_namespace = false
      chart            = "${path.module}/helm-values/efs"
      chart_version    = "0.0.1"
      values = [
        <<-EOT
          pv:
            name: efs-persist-shared
            volumeHandle: ${aws_efs_file_system.efs[count.index].id}::${aws_efs_access_point.efs_shared_ap[count.index].id}
          pvc:
            name: efs-persist-shared
        EOT
      ]
    }
  }

  depends_on = [kubernetes_namespace.jupyterhub]
}
#---------------------------------------------------------------
# Additional Resources
#---------------------------------------------------------------
resource "kubernetes_secret_v1" "huggingface_token" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = "hf-token"
    namespace = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
  }

  data = {
    token = var.huggingface_token
  }
}

resource "kubernetes_config_map_v1" "notebook" {
  count = var.enable_jupyterhub ? 1 : 0
  metadata {
    name      = "notebook"
    namespace = kubernetes_namespace.jupyterhub[count.index].metadata[0].name
  }
}
