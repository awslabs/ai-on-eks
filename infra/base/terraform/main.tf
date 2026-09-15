provider "aws" {
  region = local.region
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = ephemeral.aws_eks_cluster_auth.this.token
}

provider "helm" {
  kubernetes = {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    token                  = ephemeral.aws_eks_cluster_auth.this.token
  }

  registries = [
    {
      url      = "oci://public.ecr.aws"
      username = ephemeral.aws_ecrpublic_authorization_token.token.user_name
      password = ephemeral.aws_ecrpublic_authorization_token.token.password
    }
  ]
}

provider "kubectl" {
  apply_retry_count      = 30
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  token                  = ephemeral.aws_eks_cluster_auth.this.token
  load_config_file       = false
  lazy_load              = true
}

ephemeral "aws_eks_cluster_auth" "this" {
  name = module.eks.cluster_name
}

ephemeral "aws_ecrpublic_authorization_token" "token" {
  region = "us-east-1"
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_iam_session_context" "current" {
  arn = data.aws_caller_identity.current.arn
}

locals {
  name                   = var.name
  partition              = data.aws_partition.current.partition
  account_id             = data.aws_caller_identity.current.account_id
  mlflow_name            = "mlflow"
  mlflow_namespace       = "mlflow"
  mlflow_service_account = "mlflow"

  tags = {
    Blueprint  = local.name
    GithubRepo = "github.com/awslabs/ai-on-eks"
  }
}

resource "random_bytes" "this" {
  length = 2
}

resource "aws_cloudformation_stack" "guidance_deployment_metrics" {
  count         = var.solution_id != null && var.solution_description != null ? 1 : 0
  name          = "tracking-stack-${random_bytes.this.hex}"
  on_failure    = "DO_NOTHING"
  template_body = <<STACK
    {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Description": "(${var.solution_id}) ${var.solution_description}",
        "Resources": {
            "EmptyResource": {
                "Type": "AWS::CloudFormation::WaitConditionHandle"
            }
        }
    }
    STACK
}
