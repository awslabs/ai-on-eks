#---------------------------------------------------------------
# Dynamo Cloud ECR Repositories
#
# This module creates ECR repositories for all Dynamo Cloud components
# and sets up the necessary environment for container builds.
# This file is copied to the _LOCAL directory during deployment.
#---------------------------------------------------------------

# Note: aws_caller_identity and aws_region data sources are defined in main.tf

# ECR Repository for Dynamo Operator
resource "aws_ecr_repository" "dynamo_operator" {
  count = var.enable_dynamo_stack ? 1 : 0
  name  = "dynamo-operator"

  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "dynamo-operator"
  }
}

# ECR Repository for Dynamo API Store
resource "aws_ecr_repository" "dynamo_api_store" {
  count = var.enable_dynamo_stack ? 1 : 0
  name  = "dynamo-api-store"

  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "dynamo-api-store"
  }
}

# ECR Repository for Dynamo Pipelines
resource "aws_ecr_repository" "dynamo_pipelines" {
  count = var.enable_dynamo_stack ? 1 : 0
  name  = "dynamo-pipelines"

  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "dynamo-pipelines"
  }
}

# ECR Repository for Dynamo Base Image
resource "aws_ecr_repository" "dynamo_base" {
  count = var.enable_dynamo_stack ? 1 : 0
  name  = "dynamo-base"

  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "dynamo-base"
  }
}

# Lifecycle policy for ECR repositories to manage image retention
resource "aws_ecr_lifecycle_policy" "dynamo_operator_policy" {
  count      = var.enable_dynamo_stack ? 1 : 0
  repository = aws_ecr_repository.dynamo_operator[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "dynamo_api_store_policy" {
  count      = var.enable_dynamo_stack ? 1 : 0
  repository = aws_ecr_repository.dynamo_api_store[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "dynamo_pipelines_policy" {
  count      = var.enable_dynamo_stack ? 1 : 0
  repository = aws_ecr_repository.dynamo_pipelines[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_ecr_lifecycle_policy" "dynamo_base_policy" {
  count      = var.enable_dynamo_stack ? 1 : 0
  repository = aws_ecr_repository.dynamo_base[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 5 base images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["latest", "v"]
          countType     = "imageCountMoreThan"
          countNumber   = 5
        }
        action = {
          type = "expire"
        }
      },
      {
        rulePriority = 2
        description  = "Delete untagged images older than 1 day"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 1
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# Note: ConfigMap with ECR repository information will be created by the Dynamo Cloud operator
# when it's deployed, using the outputs from this terraform configuration
