#---------------------------------------------------------------
# ECR IRSA Role for Dynamo Cloud
#
# This module creates an IAM role for service accounts (IRSA) that provides
# ECR access to Dynamo operator and image builder pods without requiring
# credential rotation.
#---------------------------------------------------------------

# IAM role for Dynamo ECR access via IRSA
resource "aws_iam_role" "dynamo_ecr_access" {
  count = var.enable_dynamo_stack ? 1 : 0
  name  = "${local.name}-dynamo-ecr-access"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRoleWithWebIdentity"
        Effect = "Allow"
        Principal = {
          Federated = module.eks.oidc_provider_arn
        }
        Condition = {
          StringEquals = {
            "${module.eks.oidc_provider}:sub" = [
              "system:serviceaccount:${kubernetes_namespace.dynamo_cloud[0].metadata[0].name}:dynamo-cloud-dynamo-operator-controller-manager",
              "system:serviceaccount:${kubernetes_namespace.dynamo_cloud[0].metadata[0].name}:dynamo-cloud-dynamo-operator-image-builder",
              "system:serviceaccount:${kubernetes_namespace.dynamo_cloud[0].metadata[0].name}:dynamo-cloud-dynamo-api-store"
            ]
            "${module.eks.oidc_provider}:aud" = "sts.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = local.tags
}

# IAM policy for Dynamo ECR access
resource "aws_iam_role_policy" "dynamo_ecr_access" {
  count = var.enable_dynamo_stack ? 1 : 0
  name  = "${local.name}-dynamo-ecr-access"
  role  = aws_iam_role.dynamo_ecr_access[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:DescribeRepositories",
          "ecr:DescribeImages",
          "ecr:BatchDeleteImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "*"
      }
    ]
  })
}

# Output the IRSA role ARN for use by install.sh
output "dynamo_ecr_role_arn" {
  description = "ARN of the IRSA role for Dynamo ECR access"
  value       = var.enable_dynamo_stack ? aws_iam_role.dynamo_ecr_access[0].arn : null
}
