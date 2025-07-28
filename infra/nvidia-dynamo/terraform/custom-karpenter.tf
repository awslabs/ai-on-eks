#---------------------------------------------------------------
# Custom Karpenter Node Pools for NVIDIA Dynamo
#
# This module creates high-priority Karpenter node pools with BuildKit support
# for Dynamo workloads. These node pools have higher weights than base addons
# and support both AL2023 (default) and Bottlerocket with user namespace config.
# This file is copied to the _LOCAL directory during deployment.
#---------------------------------------------------------------

# Note: EKS cluster and Karpenter data sources are available from main.tf and addons.tf

#---------------------------------------------------------------
# Custom C7i CPU Node Class with BuildKit Support
#---------------------------------------------------------------
resource "kubectl_manifest" "dynamo_c7i_cpu_nodeclass" {
  count = var.enable_custom_karpenter_nodepools ? 1 : 0

  yaml_body = yamlencode({
    apiVersion = "karpenter.k8s.aws/v1"
    kind       = "EC2NodeClass"
    metadata = {
      name = "dynamo-c7i-cpu-nodeclass"
    }
    spec = merge({
      amiFamily = local.ami_family
      role      = data.aws_iam_role.karpenter_node_role.name
      subnetSelectorTerms = [
        {
          tags = {
            "karpenter.sh/discovery" = module.eks.cluster_name
          }
        }
      ]
      securityGroupSelectorTerms = [
        {
          tags = {
            "Name" = "${module.eks.cluster_name}-node"
          }
        }
      ]
      blockDeviceMappings = var.use_bottlerocket ? [
        {
          deviceName = "/dev/xvda"
          ebs = {
            volumeSize = "100Gi"
            volumeType = "gp3"
            encrypted  = true
          }
        },
        {
          deviceName = "/dev/xvdb"
          ebs = {
            volumeSize = "300Gi"
            volumeType = "gp3"
            encrypted  = true
          }
        }
        ] : [
        {
          deviceName = "/dev/xvda"
          ebs = {
            volumeSize = "100Gi"
            volumeType = "gp3"
            encrypted  = true
          }
        }
      ]
      tags = merge(local.custom_karpenter_tags, {
        Name        = "dynamo-c7i-cpu-karpenter"
        Environment = "dynamo"
        AMIFamily   = local.ami_family
      })
      },
    local.ami_config)
  })

  depends_on = [module.eks_blueprints_addons]
}


resource "kubectl_manifest" "dynamo_c7i_cpu_nodepool" {
  count = var.enable_custom_karpenter_nodepools ? 1 : 0

  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata = {
      name = "dynamo-c7i-cpu-nodepool"
    }
    spec = {
      template = {
        metadata = {
          labels = {
            "dynamo.ai/node-type"           = "c7i-cpu"
            "dynamo.ai/buildkit-compatible" = "true"
            "type"                          = "karpenter"
            "instanceType"                  = "dynamo-c7i-cpu"
          }
        }
        spec = {
          nodeClassRef = {
            group      = "karpenter.k8s.aws"
            apiVersion = "karpenter.k8s.aws/v1"
            kind       = "EC2NodeClass"
            name       = "dynamo-c7i-cpu-nodeclass"
          }
          requirements = [
            {
              key      = "karpenter.k8s.aws/instance-family"
              operator = "In"
              values   = ["c7i"]
            },
            {
              key      = "karpenter.k8s.aws/instance-size"
              operator = "In"
              values   = ["large", "xlarge", "2xlarge", "4xlarge", "8xlarge", "12xlarge", "16xlarge", "24xlarge", "48xlarge"]
            },
            {
              key      = "kubernetes.io/arch"
              operator = "In"
              values   = ["amd64"]
            },
            {
              key      = "karpenter.sh/capacity-type"
              operator = "In"
              values   = ["spot", "on-demand"]
            }
          ]
        }
      }
      limits = {
        cpu    = var.karpenter_cpu_limits
        memory = "${var.karpenter_memory_limits}Gi"
      }
      disruption = {
        consolidationPolicy = "WhenEmpty"
        consolidateAfter    = "300s"
        expireAfter         = "720h"
      }
      # Higher weight than base addons for Dynamo workload priority (base=50, dynamo=100)
      weight = 100
    }
  })

  depends_on = [kubectl_manifest.dynamo_c7i_cpu_nodeclass]
}

#---------------------------------------------------------------
# Custom G6 GPU Node Pool with Bottlerocket User Namespaces
#---------------------------------------------------------------
resource "kubectl_manifest" "dynamo_g6_gpu_nodeclass" {
  count = var.enable_custom_karpenter_nodepools ? 1 : 0

  yaml_body = yamlencode({
    apiVersion = "karpenter.k8s.aws/v1"
    kind       = "EC2NodeClass"
    metadata = {
      name = "dynamo-g6-gpu-nodeclass"
    }
    spec = merge({
      amiFamily = local.ami_family
      role      = data.aws_iam_role.karpenter_node_role.name
      subnetSelectorTerms = [
        {
          tags = {
            "karpenter.sh/discovery" = module.eks.cluster_name
          }
        }
      ]
      securityGroupSelectorTerms = [
        {
          tags = {
            "Name" = "${module.eks.cluster_name}-node"
          }
        }
      ]
      instanceStorePolicy = "RAID0"
      blockDeviceMappings = var.use_bottlerocket ? [
        {
          deviceName = "/dev/xvda"
          ebs = {
            volumeSize = "100Gi"
            volumeType = "gp3"
            encrypted  = true
          }
        },
        {
          deviceName = "/dev/xvdb"
          ebs = {
            volumeSize = "500Gi"
            volumeType = "gp3"
            encrypted  = true
          }
        }
        ] : [
        {
          deviceName = "/dev/xvda"
          ebs = {
            volumeSize = "100Gi"
            volumeType = "gp3"
            encrypted  = true
          }
        }
      ]
      tags = merge(local.custom_karpenter_tags, {
        Name        = "dynamo-g6-gpu-karpenter"
        Environment = "dynamo"
        AMIFamily   = local.ami_family
      })
      },
    local.ami_config)
  })

  depends_on = [module.eks_blueprints_addons]
}

resource "kubectl_manifest" "dynamo_g6_gpu_nodepool" {
  count = var.enable_custom_karpenter_nodepools ? 1 : 0

  yaml_body = yamlencode({
    apiVersion = "karpenter.sh/v1"
    kind       = "NodePool"
    metadata = {
      name = "dynamo-g6-gpu-nodepool"
    }
    spec = {
      template = {
        metadata = {
          labels = {
            "dynamo.ai/node-type"           = "g6-gpu"
            "dynamo.ai/buildkit-compatible" = "true"
            "type"                          = "karpenter"
            "instanceType"                  = "dynamo-g6-gpu"
            "accelerator"                   = "nvidia"
            "gpuType"                       = "l4"
          }
        }
        spec = {
          nodeClassRef = {
            group      = "karpenter.k8s.aws"
            apiVersion = "karpenter.k8s.aws/v1"
            kind       = "EC2NodeClass"
            name       = "dynamo-g6-gpu-nodeclass"
          }
          requirements = [
            {
              key      = "karpenter.k8s.aws/instance-family"
              operator = "In"
              values   = ["g6"]
            },
            {
              key      = "karpenter.k8s.aws/instance-size"
              operator = "In"
              values   = ["large", "xlarge", "2xlarge", "4xlarge", "8xlarge", "12xlarge", "16xlarge", "24xlarge", "48xlarge"]
            },
            {
              key      = "kubernetes.io/arch"
              operator = "In"
              values   = ["amd64"]
            },
            {
              key      = "karpenter.sh/capacity-type"
              operator = "In"
              values   = ["spot", "on-demand"]
            }
          ]
          taints = [
            {
              key    = "nvidia.com/gpu"
              value  = "true"
              effect = "NoSchedule"
            }
          ]
        }
      }
      limits = {
        cpu    = var.karpenter_cpu_limits
        memory = "${var.karpenter_memory_limits}Gi"
      }
      disruption = {
        consolidationPolicy = "WhenEmpty"
        consolidateAfter    = "300s"
        expireAfter         = "720h"
      }
      # Higher weight than base addons for Dynamo workload priority (base=50, dynamo=100)
      weight = 100
    }
  })

  depends_on = [kubectl_manifest.dynamo_g6_gpu_nodeclass]
}
