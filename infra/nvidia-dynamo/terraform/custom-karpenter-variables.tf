#---------------------------------------------------------------
# Variables for Custom Karpenter Node Pools
#
# This module defines variables for high-priority Karpenter node pools
# with BuildKit support for Dynamo workloads.
# This file is copied to the _LOCAL directory during deployment.
#---------------------------------------------------------------

# Note: cluster_name and karpenter_node_iam_role_arn are available from base terraform

variable "karpenter_cpu_limits" {
  description = "CPU limits for Karpenter node pools"
  type        = number
  default     = 1000
}

variable "karpenter_memory_limits" {
  description = "Memory limits for Karpenter node pools (in Gi)"
  type        = number
  default     = 1000
}

variable "enable_custom_karpenter_nodepools" {
  description = "Enable custom Karpenter node pools with user namespace support"
  type        = bool
  default     = false
}

variable "use_bottlerocket" {
  description = "Use Bottlerocket AMI family instead of AL2023. When enabled, includes max_user_namespaces configuration for buildkit compatibility."
  type        = bool
  default     = false
}

#---------------------------------------------------------------
# Data sources for custom Karpenter node pools
#---------------------------------------------------------------

# Get Karpenter node IAM role from base infrastructure
data "aws_iam_role" "karpenter_node_role" {
  name = split("/", module.eks_blueprints_addons.karpenter.node_iam_role_arn)[1]
}

#---------------------------------------------------------------
# Local values for custom node pools
#---------------------------------------------------------------
locals {
  # AMI family selection
  ami_family = var.use_bottlerocket ? "Bottlerocket" : "AL2023"

  # User data for Bottlerocket with user namespace configuration
  bottlerocket_user_data = <<-EOT
    [settings.kernel.sysctl]
    "user.max_user_namespaces" = "16384"

    [settings]
    motd = "Dynamo Custom Node - User Namespaces Enabled for BuildKit"
  EOT

  # User data for AL2023 (user namespaces enabled by default)
  al2023_user_data = <<-EOT
    #!/bin/bash
    # AL2023 user data - user namespaces enabled by default
    echo "Dynamo AL2023 Custom Node - Ready for BuildKit"
  EOT

  # Common tags for all custom Karpenter resources
  custom_karpenter_tags = {
    "dynamo.ai/managed-by" = "terraform"
    "dynamo.ai/component"  = "karpenter"
    "dynamo.ai/buildkit"   = "enabled"
  }

  # Conditional AMI and user data configuration
  ami_config = var.use_bottlerocket ? {
    amiSelectorTerms = [
      {
        alias = "bottlerocket@latest"
      }
    ]
    userData = base64encode(local.bottlerocket_user_data)
    } : {
    amiSelectorTerms = [
      {
        alias = "al2023@latest"
      }
    ]
    userData = base64encode(local.al2023_user_data)
  }
}
