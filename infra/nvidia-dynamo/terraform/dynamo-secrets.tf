#---------------------------------------------------------------
# Dynamo Cloud Kubernetes Secrets
#
# This module creates Kubernetes secrets required for Dynamo Cloud
# deployment, including Docker registry credentials for ECR access.
# This file is copied to the _LOCAL directory during deployment.
#---------------------------------------------------------------

# Data source to get ECR authorization token
data "aws_ecr_authorization_token" "token" {
  count       = var.enable_dynamo_stack ? 1 : 0
  registry_id = data.aws_caller_identity.current.account_id
}

# Kubernetes namespace for Dynamo Cloud
resource "kubernetes_namespace" "dynamo_cloud" {
  count = var.enable_dynamo_stack ? 1 : 0

  metadata {
    name = "dynamo-cloud"

    labels = {
      "app.kubernetes.io/name"       = "dynamo-cloud"
      "app.kubernetes.io/component"  = "namespace"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  depends_on = [module.eks]
}

# Docker registry secret for ECR access
resource "kubernetes_secret" "docker_registry" {
  count = var.enable_dynamo_stack ? 1 : 0

  metadata {
    name      = "docker-imagepullsecret"
    namespace = kubernetes_namespace.dynamo_cloud[0].metadata[0].name

    labels = {
      "app.kubernetes.io/name"       = "dynamo-cloud"
      "app.kubernetes.io/component"  = "docker-registry-secret"
      "app.kubernetes.io/managed-by" = "terraform"
    }
  }

  type = "kubernetes.io/dockerconfigjson"

  data = {
    ".dockerconfigjson" = jsonencode({
      auths = {
        "${data.aws_caller_identity.current.account_id}.dkr.ecr.${local.region}.amazonaws.com" = {
          "username" = "AWS"
          "password" = data.aws_ecr_authorization_token.token[0].password
          "auth"     = base64encode("AWS:${data.aws_ecr_authorization_token.token[0].password}")
        }
      }
    })
  }

  depends_on = [
    kubernetes_namespace.dynamo_cloud,
    aws_ecr_repository.dynamo_operator,
    aws_ecr_repository.dynamo_api_store,
    aws_ecr_repository.dynamo_pipelines,
    aws_ecr_repository.dynamo_base
  ]
}
