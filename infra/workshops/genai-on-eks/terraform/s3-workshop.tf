################################################################################
# S3 Bucket for Model Storage
################################################################################

locals {
  workshop_bucket_name = "genai-models-${data.aws_caller_identity.current.account_id}"
}

resource "aws_s3_bucket" "model_storage" {
  bucket        = local.workshop_bucket_name
  force_destroy = true

  tags = {
    Name        = local.workshop_bucket_name
    Purpose     = "ML Model Storage"
    Environment = "workshop"
    CostCenter  = "genai-workshop"
  }
}

# Configure bucket versioning
resource "aws_s3_bucket_versioning" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Configure server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Configure lifecycle policy
resource "aws_s3_bucket_lifecycle_configuration" "model_storage" {
  bucket = aws_s3_bucket.model_storage.id

  rule {
    id     = "model_storage_lifecycle"
    status = "Enabled"

    # Transition to Infrequent Access after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    # Clean up incomplete multipart uploads after 7 days
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Outputs moved to outputs.tf for better organization

# Kubernetes Service Account
resource "kubectl_manifest" "s3_sync_service_account" {
  depends_on = [
    module.eks
  ]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ServiceAccount
    metadata:
      name: model-storage-sa
      namespace: default
  YAML
}

# S3 CSI Driver
resource "aws_eks_addon" "s3_csi_driver" {
  cluster_name                = module.eks.cluster_name
  addon_name                  = "aws-mountpoint-s3-csi-driver"
  resolve_conflicts_on_create = "OVERWRITE"
  resolve_conflicts_on_update = "OVERWRITE"
}

resource "aws_iam_role" "s3_csi_driver_role" {
  name_prefix = "${local.name}-s3-csi-"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "pods.eks.amazonaws.com" }, Action = ["sts:AssumeRole", "sts:TagSession"] }]
  })
}

resource "aws_iam_policy" "s3_csi_driver_policy" {
  name_prefix = "${local.name}-s3-csi-"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = "arn:aws:s3:::${aws_s3_bucket.model_storage.bucket}" },
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"], Resource = "arn:aws:s3:::${aws_s3_bucket.model_storage.bucket}/*" }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "s3_csi_driver" {
  role       = aws_iam_role.s3_csi_driver_role.name
  policy_arn = aws_iam_policy.s3_csi_driver_policy.arn
}

resource "aws_eks_pod_identity_association" "s3_csi_controller" {
  cluster_name    = module.eks.cluster_name
  namespace       = "kube-system"
  service_account = "s3-csi-driver-controller-sa"
  role_arn        = aws_iam_role.s3_csi_driver_role.arn
  depends_on      = [aws_eks_addon.s3_csi_driver]
}

resource "aws_eks_pod_identity_association" "s3_csi" {
  cluster_name    = module.eks.cluster_name
  namespace       = "kube-system"
  service_account = "s3-csi-driver-sa"
  role_arn        = aws_iam_role.s3_csi_driver_role.arn
  depends_on      = [aws_eks_addon.s3_csi_driver]
}

################################################################################
# Kubernetes S3 CSI Storage Resources
################################################################################

# Create S3 prefix/folder for models
resource "aws_s3_object" "model_prefix" {
  bucket = aws_s3_bucket.model_storage.bucket
  key    = "mistral-7b-v0-3/"
  source = "/dev/null"
}

# PersistentVolume for S3 model storage
resource "kubectl_manifest" "mistral_model_pv" {
  depends_on = [
    module.eks,
    aws_s3_bucket.model_storage,
    aws_s3_object.model_prefix,
    aws_eks_addon.s3_csi_driver
  ]

  yaml_body = yamlencode({
    apiVersion = "v1"
    kind       = "PersistentVolume"
    metadata = {
      name = "mistral-model-pv"
    }
    spec = {
      capacity = {
        storage = "20Gi"
      }
      accessModes                   = ["ReadOnlyMany"]
      persistentVolumeReclaimPolicy = "Retain"
      storageClassName              = ""
      csi = {
        driver       = "s3.csi.aws.com"
        volumeHandle = "s3-csi-driver-volume"
        volumeAttributes = {
          bucketName = aws_s3_bucket.model_storage.bucket
          prefix     = "mistral-7b-v0-3/"
        }
      }
    }
  })
}

# PersistentVolumeClaim for model access
resource "kubectl_manifest" "mistral_model_pvc" {
  depends_on = [
    module.eks,
    kubectl_manifest.mistral_model_pv
  ]

  yaml_body = yamlencode({
    apiVersion = "v1"
    kind       = "PersistentVolumeClaim"
    metadata = {
      name      = "mistral-model-pvc"
      namespace = "default"
    }
    spec = {
      accessModes      = ["ReadOnlyMany"]
      volumeName       = "mistral-model-pv"
      storageClassName = ""
      resources = {
        requests = {
          storage = "20Gi"
        }
      }
    }
  })
}
