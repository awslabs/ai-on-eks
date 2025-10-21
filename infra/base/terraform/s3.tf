#---------------------------------------------------------------
# S3 Model Storage
#---------------------------------------------------------------
locals {
  s3_models_bucket_name = var.enable_s3_models_storage ? (
    var.s3_models_bucket_create ?
    aws_s3_bucket.models_bucket[0].id :
    var.s3_models_bucket_name
  ) : ""

  s3_models_primary_bucket_resources = var.enable_s3_models_storage && local.s3_models_bucket_name != "" ? [
    "arn:aws:s3:::${local.s3_models_bucket_name}",
    "arn:aws:s3:::${local.s3_models_bucket_name}/*"
  ] : []

  s3_models_additional_bucket_resources = var.enable_s3_models_storage ? flatten([
    for bucket in var.s3_models_additional_buckets : [
      "arn:aws:s3:::${bucket}",
      "arn:aws:s3:::${bucket}/*"
    ]
  ]) : []

  s3_models_all_bucket_resources = var.enable_s3_models_storage ? concat(
    local.s3_models_primary_bucket_resources,
    local.s3_models_additional_bucket_resources
  ) : []
}

# S3 bucket for storing ML models
resource "aws_s3_bucket" "models_bucket" {
  count = var.enable_s3_models_storage && var.s3_models_bucket_create ? 1 : 0

  bucket        = var.s3_models_bucket_name != "" ? var.s3_models_bucket_name : null
  bucket_prefix = var.s3_models_bucket_name == "" ? "${local.name}-models-${local.region}-" : null

  tags = merge(local.tags, {
    Purpose = "ML Model Storage"
  })
}

# Configure server-side encryption with AWS managed KMS key
resource "aws_s3_bucket_server_side_encryption_configuration" "models_bucket_encryption" {
  count = var.enable_s3_models_storage && var.s3_models_bucket_create ? 1 : 0

  bucket = aws_s3_bucket.models_bucket[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = "alias/aws/s3"
    }
    bucket_key_enabled = true
  }
}

# Set up public access blocking for security
resource "aws_s3_bucket_public_access_block" "models_bucket_pab" {
  count = var.enable_s3_models_storage && var.s3_models_bucket_create ? 1 : 0

  bucket = aws_s3_bucket.models_bucket[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM Role for Model Sync Operations
resource "aws_iam_role" "model_sync_role" {
  count = var.enable_s3_models_storage ? 1 : 0

  name_prefix = "${local.name}-model-sync-role-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })

  tags = merge(local.tags, {
    Name    = "${local.name}-model-sync-role"
    Purpose = "S3 Model Upload Operations"
  })
}

# IAM Policy for Model Sync Operations
resource "aws_iam_policy" "model_sync_policy" {
  count = var.enable_s3_models_storage ? 1 : 0

  name_prefix = "${local.name}-model-sync-policy-"
  description = "Policy for S3 model upload operations"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = local.s3_models_all_bucket_resources
      }
    ]
  })

  tags = merge(local.tags, {
    Name    = "${local.name}-model-sync-policy"
    Purpose = "S3 Model Upload Operations"
  })
}

# Attach policy to sync role
resource "aws_iam_role_policy_attachment" "model_sync_policy_attachment" {
  count = var.enable_s3_models_storage ? 1 : 0

  role       = aws_iam_role.model_sync_role[0].name
  policy_arn = aws_iam_policy.model_sync_policy[0].arn
}

# IAM Role for Model Inference Operations
resource "aws_iam_role" "model_inference_role" {
  count = var.enable_s3_models_storage ? 1 : 0

  name_prefix = "${local.name}-model-inference-role-"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "pods.eks.amazonaws.com"
        }
        Action = [
          "sts:AssumeRole",
          "sts:TagSession"
        ]
      }
    ]
  })

  tags = merge(local.tags, {
    Name    = "${local.name}-model-inference-role"
    Purpose = "S3 Model Inference Operations"
  })
}

# IAM Policy for Model Inference Operations
resource "aws_iam_policy" "model_inference_policy" {
  count = var.enable_s3_models_storage ? 1 : 0

  name_prefix = "${local.name}-model-inference-policy-"
  description = "Policy for S3 model inference operations (read-only)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = local.s3_models_all_bucket_resources
      }
    ]
  })

  tags = merge(local.tags, {
    Name    = "${local.name}-model-inference-policy"
    Purpose = "S3 Model Inference Operations"
  })
}

# Attach policy to inference role
resource "aws_iam_role_policy_attachment" "model_inference_policy_attachment" {
  count = var.enable_s3_models_storage ? 1 : 0

  role       = aws_iam_role.model_inference_role[0].name
  policy_arn = aws_iam_policy.model_inference_policy[0].arn
}

# EKS Pod Identity Association for Sync Service Account
resource "aws_eks_pod_identity_association" "model_sync_pod_identity" {
  count = var.enable_s3_models_storage ? 1 : 0

  cluster_name    = module.eks.cluster_name
  namespace       = var.s3_models_sync_sa_namespace
  service_account = var.s3_models_sync_sa
  role_arn        = aws_iam_role.model_sync_role[0].arn

  tags = merge(local.tags, {
    Name    = "${local.name}-model-sync-pod-identity"
    Purpose = "Pod Identity for S3 Model Upload Operations"
  })
}

# EKS Pod Identity Association for Inference Service Account
resource "aws_eks_pod_identity_association" "model_inference_pod_identity" {
  count = var.enable_s3_models_storage ? 1 : 0

  cluster_name    = module.eks.cluster_name
  namespace       = var.s3_models_inference_sa_namespace
  service_account = var.s3_models_inference_sa
  role_arn        = aws_iam_role.model_inference_role[0].arn

  tags = merge(local.tags, {
    Name    = "${local.name}-model-inference-pod-identity"
    Purpose = "Pod Identity for S3 Model Inference Operations"
  })
}
