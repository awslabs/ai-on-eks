#---------------------------------------------------------------
# FSx for Lustre File system Static provisioning
#    1> Create Fsx for Lustre filesystem (Lustre FS storage capacity must be 1200, 2400, or a multiple of 3600)
#    2> Create Storage Class for Filesystem (Cluster scoped)
#    3> Persistent Volume with  Hardcoded reference to Fsx for Lustre filesystem with filesystem_id and dns_name (Cluster scoped)
#    4> Persistent Volume claim for this persistent volume will always use the same file system (Namespace scoped)
#---------------------------------------------------------------
# NOTE: FSx for Lustre file system creation can take up to 10 mins
resource "aws_fsx_lustre_file_system" "this" {
  count           = var.deploy_fsx_volume ? 1 : 0
  deployment_type = "PERSISTENT_2"
  storage_type    = "SSD"
  # EFA (RDMA) mode requires the 1000 MB/s/TiB tier; default (non-EFA) keeps 500.
  per_unit_storage_throughput = var.fsx_efa_enabled ? "1000" : "500" # 125, 250, 500, 1000
  # null => 4800 (EFA minimum) when EFA is on, else the original 2400. Larger EFA
  # capacities add OSTs (e.g. 19200 = 4 OSTs) to parallelize model-weight loading.
  storage_capacity = var.fsx_storage_capacity != null ? var.fsx_storage_capacity : (var.fsx_efa_enabled ? 4800 : 2400)

  # EFA (RDMA) transport for FSx-Lustre clients. Immutable (create-only).
  efa_enabled = var.fsx_efa_enabled

  # EFA-enabled filesystems require a metadata configuration.
  dynamic "metadata_configuration" {
    for_each = var.fsx_efa_enabled ? [1] : []
    content {
      mode = "AUTOMATIC"
    }
  }

  subnet_ids         = [module.vpc.private_subnets[0]]
  security_group_ids = [aws_security_group.fsx[0].id]
  log_configuration {
    level = "WARN_ERROR"
  }
  tags = merge({ "Name" : "${local.name}-static" }, local.tags)
}

# This process can take up to 20 mins
resource "aws_fsx_data_repository_association" "this" {
  count                = var.deploy_fsx_volume ? 1 : 0
  file_system_id       = aws_fsx_lustre_file_system.this[0].id
  data_repository_path = "s3://${module.fsx_s3_bucket[0].s3_bucket_id}"
  file_system_path     = "/data" # This directory will be used in Spark podTemplates under volumeMounts as subPath

  s3 {
    auto_export_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }

    auto_import_policy {
      events = ["NEW", "CHANGED", "DELETED"]
    }
  }

  timeouts {
    create = "20m"
  }
}

#---------------------------------------------------------------
# Sec group for FSx for Lustre
#---------------------------------------------------------------
resource "aws_security_group" "fsx" {
  count       = var.deploy_fsx_volume ? 1 : 0
  name        = "${local.name}-fsx"
  description = "Allow inbound traffic from private subnets of the VPC to FSx filesystem"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description = "Allows Lustre traffic between Lustre clients"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
    from_port   = 1021
    to_port     = 1023
    protocol    = "tcp"
  }
  ingress {
    description = "Allows Lustre traffic between Lustre clients"
    cidr_blocks = module.vpc.private_subnets_cidr_blocks
    from_port   = 988
    to_port     = 988
    protocol    = "tcp"
  }

  # EFA (RDMA) requires all-protocol self-referencing ingress/egress between the
  # FSx ENIs and clients, plus all-protocol ingress from the client subnets.
  # Added only when fsx_efa_enabled = true; the default path is unchanged.
  dynamic "ingress" {
    for_each = var.fsx_efa_enabled ? [1] : []
    content {
      description = "EFA all-protocol ingress from private subnets (FSx over EFA)"
      cidr_blocks = module.vpc.private_subnets_cidr_blocks
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
    }
  }
  dynamic "ingress" {
    for_each = var.fsx_efa_enabled ? [1] : []
    content {
      description = "EFA all-protocol self-referencing ingress (FSx over EFA)"
      self        = true
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
    }
  }
  dynamic "egress" {
    for_each = var.fsx_efa_enabled ? [1] : []
    content {
      description = "EFA all-protocol self-referencing egress (FSx over EFA)"
      self        = true
      from_port   = 0
      to_port     = 0
      protocol    = "-1"
    }
  }

  # EFA on: tag the FSx SG for Karpenter discovery so EFA GPU nodes attach it via the shared
  # EC2NodeClass securityGroupSelectorTerms (karpenter.sh/discovery) -> node<->FSx EFA/RDMA both ways.
  tags = merge(local.tags, var.fsx_efa_enabled ? { "karpenter.sh/discovery" = module.eks.cluster_name } : {})
}

#---------------------------------------------------------------
# S3 bucket for DataSync between FSx for Lustre and S3 Bucket
#---------------------------------------------------------------
#tfsec:ignore:aws-s3-enable-bucket-logging tfsec:ignore:aws-s3-enable-versioning
module "fsx_s3_bucket" {
  count   = var.deploy_fsx_volume ? 1 : 0
  source  = "terraform-aws-modules/s3-bucket/aws"
  version = "~> 5.7"

  create_bucket = true

  bucket_prefix = "${local.name}-fsx-"
  # For example only - please evaluate for your environment
  force_destroy = true

  server_side_encryption_configuration = {
    rule = {
      apply_server_side_encryption_by_default = {
        sse_algorithm = "AES256"
      }
    }
  }
  tags = local.tags
}

#---------------------------------------------------------------
# Storage Class - FSx for Lustre
#---------------------------------------------------------------
resource "kubernetes_storage_class_v1" "fsx" {
  count = var.deploy_fsx_volume ? 1 : 0
  metadata {
    name = "fsx"
  }

  storage_provisioner = "fsx.csi.aws.com"
  parameters = {
    subnetId         = module.vpc.private_subnets[0]
    securityGroupIds = aws_security_group.fsx[0].id
  }

  depends_on = [
    aws_eks_addon.aws_efs_csi_driver,
    module.efs
  ]
}

#---------------------------------------------------------------
# FSx for Lustre Persistent Volume - Static provisioning
#---------------------------------------------------------------

resource "kubernetes_persistent_volume_v1" "fsx_static_pv" {
  count = var.deploy_fsx_volume ? 1 : 0
  metadata {
    name = "fsx-static-pv"
  }
  spec {
    access_modes       = ["ReadWriteMany"]
    capacity           = { storage : "1000Gi" }
    volume_mode        = "Filesystem"
    storage_class_name = "fsx"
    persistent_volume_source {
      csi {
        driver        = "fsx.csi.aws.com"
        volume_handle = aws_fsx_lustre_file_system.this[0].id
        volume_attributes = {
          dnsname = aws_fsx_lustre_file_system.this[0].dns_name
          mountname : aws_fsx_lustre_file_system.this[0].mount_name
        }
      }
    }
  }
  depends_on = [
    aws_eks_addon.aws_fsx_csi_driver,
    kubernetes_storage_class_v1.fsx,
    aws_fsx_lustre_file_system.this
  ]
}

#---------------------------------------------------------------
# FSx for Lustre Persistent Volume Claim
#---------------------------------------------------------------
resource "kubernetes_namespace_v1" "fsx_namespace" {
  count = var.deploy_fsx_volume && var.fsx_pvc_namespace != "default" ? 1 : 0
  metadata {
    name = var.fsx_pvc_namespace
  }
  depends_on = [
    aws_eks_addon.aws_fsx_csi_driver
  ]
}

resource "kubernetes_persistent_volume_claim_v1" "fsx" {
  count = var.deploy_fsx_volume ? 1 : 0
  metadata {
    name      = "fsx-static-pvc"
    namespace = var.fsx_pvc_namespace
  }
  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = "fsx"
    volume_name        = "fsx-static-pv"
    resources {
      requests = {
        storage = "1000Gi"
      }
    }
  }
  depends_on = [
    kubernetes_persistent_volume_v1.fsx_static_pv
  ]
}
