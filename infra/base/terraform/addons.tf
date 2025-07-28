# Use this data source to get the ARN of a certificate in AWS Certificate Manager (ACM)
data "aws_acm_certificate" "issued" {
  count    = var.jupyter_hub_auth_mechanism != "dummy" ? 1 : 0
  domain   = var.acm_certificate_domain
  statuses = ["ISSUED"]
}

locals {
  cognito_custom_domain = var.cognito_custom_domain
}

#---------------------------------------------------------------
# GP3 Encrypted Storage Class
#---------------------------------------------------------------
resource "kubernetes_annotations" "disable_gp2" {
  annotations = {
    "storageclass.kubernetes.io/is-default-class" : "false"
  }
  api_version = "storage.k8s.io/v1"
  kind        = "StorageClass"
  metadata {
    name = "gp2"
  }
  force = true

  depends_on = [module.eks.eks_cluster_id]
}

resource "kubernetes_storage_class" "default_gp3" {
  metadata {
    name = "gp3"
    annotations = {
      "storageclass.kubernetes.io/is-default-class" : "true"
    }
  }

  storage_provisioner    = "ebs.csi.aws.com"
  reclaim_policy         = "Delete"
  allow_volume_expansion = true
  volume_binding_mode    = "WaitForFirstConsumer"
  parameters = {
    fsType    = "ext4"
    encrypted = true
    type      = "gp3"
  }

  depends_on = [kubernetes_annotations.disable_gp2]
}

#---------------------------------------------------------------
# Karpenter Node instance role Access Entry
#---------------------------------------------------------------
resource "aws_eks_access_entry" "karpenter_nodes" {
  cluster_name  = module.eks.cluster_name
  principal_arn = module.eks_blueprints_addons.karpenter.node_iam_role_arn
  type          = "EC2_LINUX"
}

#---------------------------------------------------------------
# EKS Blueprints Addons
#---------------------------------------------------------------
module "eks_blueprints_addons" {
  source  = "aws-ia/eks-blueprints-addons/aws"
  version = "~> 1.20"

  cluster_name      = module.eks.cluster_name
  cluster_endpoint  = module.eks.cluster_endpoint
  cluster_version   = module.eks.cluster_version
  oidc_provider_arn = module.eks.oidc_provider_arn

  enable_aws_efs_csi_driver = var.enable_aws_efs_csi_driver
  #---------------------------------------
  # AWS Load Balancer Controller Add-on
  #---------------------------------------
  enable_aws_load_balancer_controller = true
  # turn off the mutating webhook for services because we are using
  # service.beta.kubernetes.io/aws-load-balancer-type: external
  aws_load_balancer_controller = {
    set = [{
      name  = "enableServiceMutatorWebhook"
      value = "false"
    }]
  }

  #---------------------------------------
  # Ingress Nginx Add-on
  #---------------------------------------
  enable_ingress_nginx = true
  ingress_nginx = {
    version = "4.12.1"
    values  = [templatefile("${path.module}/helm-values/ingress-nginx-values.yaml", {})]
  }

  #---------------------------------------
  # Karpenter Autoscaler for EKS Cluster
  #---------------------------------------
  enable_karpenter                  = true
  karpenter_enable_spot_termination = true
  karpenter_node = {
    iam_role_additional_policies = {
      AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    }
  }
  karpenter = {
    chart_version       = "1.4.0"
    repository_username = data.aws_ecrpublic_authorization_token.token.user_name
    repository_password = data.aws_ecrpublic_authorization_token.token.password
    source_policy_documents = [
      data.aws_iam_policy_document.karpenter_controller_policy.json
    ]
  }

  #---------------------------------------
  # Argo Workflows & Argo Events
  #---------------------------------------
  enable_argo_workflows = var.enable_argo_workflows
  argo_workflows = {
    name       = "argo-workflows"
    namespace  = "argo-workflows"
    repository = "https://argoproj.github.io/argo-helm"
    values     = [templatefile("${path.module}/helm-values/argo-workflows-values.yaml", {})]
  }

  enable_argo_events = var.enable_argo_events
  argo_events = {
    name       = "argo-events"
    namespace  = "argo-events"
    repository = "https://argoproj.github.io/argo-helm"
    values     = [templatefile("${path.module}/helm-values/argo-events-values.yaml", {})]
  }

  enable_argocd = var.enable_argocd
  argocd = {
    name       = "argocd"
    namespace  = "argocd"
    repository = "https://argoproj.github.io/argo-helm"
    values     = [templatefile("${path.module}/helm-values/argocd-values.yaml", {})]
  }

  #---------------------------------------
  # Prometheus and Grafana stack
  #---------------------------------------
  #---------------------------------------------------------------
  # 1- Grafana port-forward `kubectl port-forward svc/kube-prometheus-stack-grafana 8080:80 -n kube-prometheus-stack`
  # 2- Grafana Admin user: admin
  # 3- Get secret name from Terrafrom output: `terraform output grafana_secret_name`
  # 3- Get admin user password: `aws secretsmanager get-secret-value --secret-id <REPLACE_WIRTH_SECRET_ID> --region $AWS_REGION --query "SecretString" --output text`
  #---------------------------------------------------------------
  enable_kube_prometheus_stack = var.enable_kube_prometheus_stack
  kube_prometheus_stack = {
    values = [
      var.enable_amazon_prometheus ? templatefile("${path.module}/helm-values/kube-prometheus-amp-enable.yaml", {
        region              = local.region
        amp_sa              = local.amp_ingest_service_account
        amp_irsa            = module.amp_ingest_irsa[0].iam_role_arn
        amp_remotewrite_url = "https://aps-workspaces.${local.region}.amazonaws.com/workspaces/${aws_prometheus_workspace.amp[0].id}/api/v1/remote_write"
        amp_url             = "https://aps-workspaces.${local.region}.amazonaws.com/workspaces/${aws_prometheus_workspace.amp[0].id}"
      }) : templatefile("${path.module}/helm-values/kube-prometheus.yaml", { storage_class_type = kubernetes_storage_class.default_gp3.id })
    ]
    chart_version = "48.1.1"
    set_sensitive = [
      {
        name  = "grafana.adminPassword"
        value = var.enable_kube_prometheus_stack ? data.aws_secretsmanager_secret_version.admin_password_version[0].secret_string : null
      }
    ],
  }

  #---------------------------------------
  # Enable FSx for Lustre CSI Driver
  #---------------------------------------
  enable_aws_fsx_csi_driver = var.enable_aws_fsx_csi_driver

  tags = local.tags

}

#---------------------------------------------------------------
# Data on EKS Kubernetes Addons
#---------------------------------------------------------------

module "data_addons" {
  source  = "aws-ia/eks-data-addons/aws"
  version = "1.37.1"

  oidc_provider_arn = module.eks.oidc_provider_arn

  #---------------------------------------------------------------
  # JupyterHub Add-on
  #---------------------------------------------------------------
  enable_jupyterhub = var.enable_jupyterhub
  jupyterhub_helm_config = var.enable_jupyterhub ? {
    values = [templatefile("${path.module}/helm-values/jupyterhub-values-${var.jupyter_hub_auth_mechanism}.yaml", {
      ssl_cert_arn                = try(data.aws_acm_certificate.issued[0].arn, "")
      jupyterdomain               = try("https://${var.jupyterhub_domain}/hub/oauth_callback", "")
      authorize_url               = var.oauth_domain != "" ? "${var.oauth_domain}/auth" : try("https://${local.cognito_custom_domain}.auth.${local.region}.amazoncognito.com/oauth2/authorize", "")
      token_url                   = var.oauth_domain != "" ? "${var.oauth_domain}/token" : try("https://${local.cognito_custom_domain}.auth.${local.region}.amazoncognito.com/oauth2/token", "")
      userdata_url                = var.oauth_domain != "" ? "${var.oauth_domain}/userinfo" : try("https://${local.cognito_custom_domain}.auth.${local.region}.amazoncognito.com/oauth2/userInfo", "")
      username_key                = try(var.oauth_username_key, "")
      client_id                   = var.oauth_jupyter_client_id != "" ? var.oauth_jupyter_client_id : try(aws_cognito_user_pool_client.user_pool_client[0].id, "")
      client_secret               = var.oauth_jupyter_client_secret != "" ? var.oauth_jupyter_client_secret : try(aws_cognito_user_pool_client.user_pool_client[0].client_secret, "")
      user_pool_id                = try(aws_cognito_user_pool.pool[0].id, "")
      identity_pool_id            = try(aws_cognito_identity_pool.identity_pool[0].id, "")
      jupyter_single_user_sa_name = kubernetes_service_account_v1.jupyterhub_single_user_sa[0].metadata[0].name
      region                      = var.region
    })]
    version = "3.2.1"
  } : null

  enable_volcano = var.enable_volcano
  #---------------------------------------
  # Kuberay Operator
  #---------------------------------------
  enable_kuberay_operator = var.enable_kuberay_operator
  kuberay_operator_helm_config = {
    version = "1.4.0"
    # Enabling Volcano as Batch scheduler for KubeRay Operator
    values = [
      <<-EOT
      batchScheduler:
        enabled: ${var.enable_volcano}
    EOT
    ]
  }

  #---------------------------------------------------------------
  # MLflow Tracking Add-on
  #---------------------------------------------------------------
  enable_mlflow_tracking = var.enable_mlflow_tracking

  mlflow_tracking_helm_config = {
    mlflow_namespace = try(kubernetes_namespace_v1.mlflow[0].metadata[0].name, local.mlflow_namespace)

    values = [
      templatefile("${path.module}/helm-values/mlflow-tracking-values.yaml", {
        mlflow_sa   = local.mlflow_service_account
        mlflow_irsa = try(module.mlflow_irsa[0].iam_role_arn, "")
        # MLflow Postgres RDS Config
        mlflow_db_username = local.mlflow_name
        mlflow_db_password = try(sensitive(aws_secretsmanager_secret_version.postgres[0].secret_string), "")
        mlflow_db_name     = try(module.db[0].db_instance_name, "")
        mlflow_db_host     = try(element(split(":", module.db[0].db_instance_endpoint), 0), "")
        # S3 bucket config for artifacts
        s3_bucket_name = try(module.mlflow_s3_bucket[0].s3_bucket_id, "")
      })
    ]
  }

  #---------------------------------------
  # EFA Device Plugin Add-on
  #---------------------------------------
  # IMPORTANT: Enable EFA only on nodes with EFA devices attached.
  # Otherwise, you'll encounter the "No devices found..." error. Restart the pod after attaching an EFA device, or use a node selector to prevent incompatible scheduling.
  enable_aws_efa_k8s_device_plugin = var.enable_aws_efa_k8s_device_plugin
  aws_efa_k8s_device_plugin_helm_config = {
    values = [file("${path.module}/helm-values/aws-efa-k8s-device-plugin-values.yaml")]
  }

  #---------------------------------------------------------------
  # Kubecost Add-on
  #---------------------------------------------------------------
  enable_kubecost = var.enable_kubecost
  kubecost_helm_config = {
    values              = [templatefile("${path.module}/helm-values/kubecost-values.yaml", {})]
    version             = "2.2.2"
    repository_username = data.aws_ecrpublic_authorization_token.token.user_name
    repository_password = data.aws_ecrpublic_authorization_token.token.password
  }

  #---------------------------------------------------------------
  # Neuron Add-on
  #---------------------------------------------------------------
  enable_aws_neuron_device_plugin = true

  aws_neuron_device_plugin_helm_config = {
    # Enable default scheduler
    values = [
      <<-EOT
      devicePlugin:
        tolerations:
        - key: CriticalAddonsOnly
          operator: Exists
        - key: aws.amazon.com/neuron
          operator: Exists
          effect: NoSchedule
        - key: hub.jupyter.org/dedicated
          operator: Exists
          effect: NoSchedule
      scheduler:
        enabled: true
      npd:
        enabled: false
      EOT
    ]
  }

  #---------------------------------------------------------------
  # Karpenter Resources Add-on
  #---------------------------------------------------------------
  enable_karpenter_resources = true
  karpenter_resources_helm_config = {
    g6-gpu-karpenter = {
      values = [
        <<-EOT
      name: g6-gpu-karpenter
      clusterName: ${module.eks.cluster_name}
      ec2NodeClass:
        amiFamily: Bottlerocket
        karpenterRole: ${split("/", module.eks_blueprints_addons.karpenter.node_iam_role_arn)[1]}
        subnetSelectorTerms:
          id: ${module.vpc.private_subnets[2]}
        securityGroupSelectorTerms:
          tags:
            Name: ${module.eks.cluster_name}-node
        instanceStorePolicy: RAID0
        blockDeviceMappings:
          # Root device
          - deviceName: /dev/xvda
            ebs:
              volumeSize: 50Gi
              volumeType: gp3
              encrypted: true

      nodePool:
        labels:
          - instanceType: g6-gpu-karpenter
          - type: karpenter
          - gpuType: l4
        taints:
          - key: nvidia.com/gpu
            value: "Exists"
            effect: "NoSchedule"
        requirements:
          - key: "karpenter.k8s.aws/instance-family"
            operator: In
            values: ["g6"]
          - key: "karpenter.k8s.aws/instance-size"
            operator: In
            values: [ "2xlarge", "4xlarge", "8xlarge", "12xlarge", "16xlarge", "24xlarge", "48xlarge" ]
          - key: "kubernetes.io/arch"
            operator: In
            values: ["amd64"]
          - key: "karpenter.sh/capacity-type"
            operator: In
            values: ["spot", "on-demand"]
        limits:
          cpu: 1000
        disruption:
          consolidationPolicy: WhenEmpty
          consolidateAfter: 300s
          expireAfter: 720h
        weight: 100
      EOT
      ]
    }
    g5-gpu-karpenter = {
      values = [
        <<-EOT
      name: g5-gpu-karpenter
      clusterName: ${module.eks.cluster_name}
      ec2NodeClass:
        amiFamily: Bottlerocket
        amiSelectorTerms:
          - alias: bottlerocket@latest
        karpenterRole: ${split("/", module.eks_blueprints_addons.karpenter.node_iam_role_arn)[1]}
        subnetSelectorTerms:
          id: ${module.vpc.private_subnets[2]}
        securityGroupSelectorTerms:
          tags:
            Name: ${module.eks.cluster_name}-node
        instanceStorePolicy: RAID0
        blockDeviceMappings:
          # Root device
          - deviceName: /dev/xvda
            ebs:
              volumeSize: 50Gi
              volumeType: gp3
              encrypted: true
          # Data device: Container resources such as images and logs
          - deviceName: /dev/xvdb
            ebs:
              volumeSize: 300Gi
              volumeType: gp3
              encrypted: true
              ${var.bottlerocket_data_disk_snapshot_id != null ? "snapshotID: ${var.bottlerocket_data_disk_snapshot_id}" : ""}

      nodePool:
        labels:
          - instanceType: g5-gpu-karpenter
          - type: karpenter
          - accelerator: nvidia
          - gpuType: a10g
        taints:
          - key: nvidia.com/gpu
            value: "Exists"
            effect: "NoSchedule"
        requirements:
          - key: "karpenter.k8s.aws/instance-family"
            operator: In
            values: ["g5"]
          - key: "karpenter.k8s.aws/instance-size"
            operator: In
            values: [ "2xlarge", "4xlarge", "8xlarge", "12xlarge", "16xlarge", "24xlarge", "48xlarge" ]
          - key: "kubernetes.io/arch"
            operator: In
            values: ["amd64"]
          - key: "karpenter.sh/capacity-type"
            operator: In
            values: ["spot", "on-demand"]
        limits:
          cpu: 1000
        disruption:
          consolidationPolicy: WhenEmpty
          consolidateAfter: 300s
          expireAfter: 720h
        weight: 100
      EOT
      ]
    }
    x86-cpu-karpenter = {
      values = [
        <<-EOT
      name: x86-cpu-karpenter
      clusterName: ${module.eks.cluster_name}
      ec2NodeClass:
        amiFamily: Bottlerocket
        amiSelectorTerms:
          - alias: bottlerocket@latest
        karpenterRole: ${split("/", module.eks_blueprints_addons.karpenter.node_iam_role_arn)[1]}
        subnetSelectorTerms:
          id: ${module.vpc.private_subnets[3]}
        securityGroupSelectorTerms:
          tags:
            Name: ${module.eks.cluster_name}-node
        blockDeviceMappings:
          # Root device
          - deviceName: /dev/xvda
            ebs:
              volumeSize: 100Gi
              volumeType: gp3
              encrypted: true
          # Data device: Container resources such as images and logs
          - deviceName: /dev/xvdb
            ebs:
              volumeSize: 300Gi
              volumeType: gp3
              encrypted: true
              ${var.bottlerocket_data_disk_snapshot_id != null ? "snapshotID: ${var.bottlerocket_data_disk_snapshot_id}" : ""}

      nodePool:
        labels:
          - type: karpenter
          - instanceType: x86-cpu-karpenter
        requirements:
          - key: "karpenter.k8s.aws/instance-family"
            operator: In
            values: ["m5"]
          - key: "karpenter.k8s.aws/instance-size"
            operator: In
            values: [ "xlarge", "2xlarge", "4xlarge", "8xlarge"]
          - key: "kubernetes.io/arch"
            operator: In
            values: ["amd64"]
          - key: "karpenter.sh/capacity-type"
            operator: In
            values: ["spot", "on-demand"]
        limits:
          cpu: 1000
        disruption:
          consolidationPolicy: WhenEmpty
          consolidateAfter: 300s
          expireAfter: 720h
        weight: 100
      EOT
      ]
    }
    trainium-trn1 = {
      values = [
        <<-EOT
      name: trainium-trn1
      clusterName: ${module.eks.cluster_name}
      ec2NodeClass:
        amiSelectorTerms:
          - alias: bottlerocket@latest
        karpenterRole: ${split("/", module.eks_blueprints_addons.karpenter.node_iam_role_arn)[1]}
        subnetSelectorTerms:
          id: ${module.vpc.private_subnets[2]}
        securityGroupSelectorTerms:
          tags:
            Name: ${module.eks.cluster_name}-node
        instanceStorePolicy: RAID0
        blockDeviceMappings:
          # Root device
          - deviceName: /dev/xvda
            ebs:
              volumeSize: 100Gi
              volumeType: gp3
              encrypted: true
          # Data device: Container resources such as images and logs
          - deviceName: /dev/xvdb
            ebs:
              volumeSize: 300Gi
              volumeType: gp3
              encrypted: true
              ${var.bottlerocket_data_disk_snapshot_id != null ? "snapshotID: ${var.bottlerocket_data_disk_snapshot_id}" : ""}

      nodePool:
        labels:
          - type: karpenter
          - instanceType: trainium-trn1
          - accelerator: neuron
        taints:
          - key: aws.amazon.com/neuron
            value: "true"
            effect: "NoSchedule"
        requirements:
          - key: "karpenter.k8s.aws/instance-family"
            operator: In
            values: ["trn1"]
          - key: "kubernetes.io/arch"
            operator: In
            values: ["amd64"]
          - key: "karpenter.sh/capacity-type"
            operator: In
            values: ["on-demand"]
        limits:
          cpu: 1000
        disruption:
          consolidationPolicy: WhenEmpty
          consolidateAfter: 300s
          expireAfter: 720h
        weight: 100
      EOT
      ]
    }
    inferentia-inf2 = {
      values = [
        <<-EOT
      name: inferentia-inf2
      clusterName: ${module.eks.cluster_name}
      ec2NodeClass:
        amiSelectorTerms:
          - alias: bottlerocket@latest
        karpenterRole: ${split("/", module.eks_blueprints_addons.karpenter.node_iam_role_arn)[1]}
        subnetSelectorTerms:
          id: ${module.vpc.private_subnets[2]}
        securityGroupSelectorTerms:
          tags:
            Name: ${module.eks.cluster_name}-node
        blockDeviceMappings:
          # Root device
          - deviceName: /dev/xvda
            ebs:
              volumeSize: 100Gi
              volumeType: gp3
              encrypted: true
          # Data device: Container resources such as images and logs
          - deviceName: /dev/xvdb
            ebs:
              volumeSize: 300Gi
              volumeType: gp3
              encrypted: true
              ${var.bottlerocket_data_disk_snapshot_id != null ? "snapshotID: ${var.bottlerocket_data_disk_snapshot_id}" : ""}
      nodePool:
        labels:
          - instanceType: inferentia-inf2
          - type: karpenter
          - accelerator: neuron
        taints:
          - key: aws.amazon.com/neuron
            value: "true"
            effect: "NoSchedule"
        requirements:
          - key: "karpenter.k8s.aws/instance-family"
            operator: In
            values: ["inf2"]
          - key: "kubernetes.io/arch"
            operator: In
            values: ["amd64"]
          - key: "karpenter.sh/capacity-type"
            operator: In
            values: [ "on-demand"]
        limits:
          cpu: 1000
        disruption:
          consolidationPolicy: WhenEmpty
          consolidateAfter: 300s
          expireAfter: 720h
        weight: 100
      EOT
      ]
    }
  }

  depends_on = [
    kubernetes_secret_v1.huggingface_token,
    kubernetes_config_map_v1.notebook
  ]
}

#---------------------------------------------------------------
# ETCD for TorchX
#---------------------------------------------------------------
data "http" "torchx_etcd_yaml" {
  url = "https://raw.githubusercontent.com/pytorch/torchx/main/resources/etcd.yaml"
}

data "kubectl_file_documents" "torchx_etcd_yaml" {
  content = data.http.torchx_etcd_yaml.response_body
}

resource "kubectl_manifest" "torchx_etcd" {
  for_each   = var.enable_torchx_etcd ? data.kubectl_file_documents.torchx_etcd_yaml.manifests : {}
  yaml_body  = each.value
  depends_on = [module.eks.eks_cluster_id]
}

#---------------------------------------------------------------
# Grafana Admin credentials resources
# Login to AWS secrets manager with the same role as Terraform to extract the Grafana admin password with the secret name as "grafana"
#---------------------------------------------------------------
data "aws_secretsmanager_secret_version" "admin_password_version" {
  count      = var.enable_kube_prometheus_stack ? 1 : 0
  secret_id  = aws_secretsmanager_secret.grafana[count.index].id
  depends_on = [aws_secretsmanager_secret_version.grafana]
}

resource "random_password" "grafana" {
  count            = var.enable_kube_prometheus_stack ? 1 : 0
  length           = 16
  special          = true
  override_special = "@_"
}

#tfsec:ignore:aws-ssm-secret-use-customer-key
resource "aws_secretsmanager_secret" "grafana" {
  count                   = var.enable_kube_prometheus_stack ? 1 : 0
  name_prefix             = "${local.name}-oss-grafana"
  recovery_window_in_days = 0 # Set to zero for this example to force delete during Terraform destroy
}

resource "aws_secretsmanager_secret_version" "grafana" {
  count         = var.enable_kube_prometheus_stack ? 1 : 0
  secret_id     = aws_secretsmanager_secret.grafana[count.index].id
  secret_string = random_password.grafana[count.index].result
}

resource "kubectl_manifest" "neuron_monitor" {
  yaml_body  = file("${path.module}/monitoring/neuron-monitor-daemonset.yaml")
  depends_on = [module.eks_blueprints_addons]
}

resource "kubectl_manifest" "efs_sc" {
  count     = var.enable_aws_efs_csi_driver ? 1 : 0
  yaml_body = <<YAML
    apiVersion: storage.k8s.io/v1
    kind: StorageClass
    metadata:
      name: efs-sc
    provisioner: efs.csi.aws.com
    mountOptions:
      - iam
  YAML
}

data "aws_iam_policy_document" "karpenter_controller_policy" {
  statement {
    actions = [
      "ec2:RunInstances",
      "ec2:CreateLaunchTemplate",
    ]
    resources = ["*"]
    effect    = "Allow"
    sid       = "KarpenterControllerAdditionalPolicy"
  }
}
#---------------------------------------------------------------
# MPI Operator for distributed training on Trainium
#---------------------------------------------------------------
data "http" "mpi_operator_yaml" {
  url = "https://raw.githubusercontent.com/kubeflow/mpi-operator/v0.4.0/deploy/v2beta1/mpi-operator.yaml"
}

data "kubectl_file_documents" "mpi_operator_yaml" {
  content = data.http.mpi_operator_yaml.response_body
}

resource "kubectl_manifest" "mpi_operator" {
  for_each   = var.enable_mpi_operator ? data.kubectl_file_documents.mpi_operator_yaml.manifests : {}
  yaml_body  = each.value
  depends_on = [module.eks.eks_cluster_id]
}
