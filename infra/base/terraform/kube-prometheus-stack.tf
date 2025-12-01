# TODO: Currently we don't do anything with AMP
locals {
  kube_prometheus_values = templatefile("${path.module}/helm-values/kube-prometheus.yaml", {
    # Add template variables if needed for AMP integration
    region              = local.region
    amp_sa              = local.amp_ingest_service_account
    amp_remotewrite_url = var.enable_amazon_prometheus ? "https://aps-workspaces.${local.region}.amazonaws.com/workspaces/${aws_prometheus_workspace.amp[0].id}/api/v1/remote_write" : ""
    amp_url             = var.enable_amazon_prometheus ? "https://aps-workspaces.${local.region}.amazonaws.com/workspaces/${aws_prometheus_workspace.amp[0].id}" : ""
    storage_class_name  = var.enable_eks_auto_mode ? "auto-ebs-sc" : "gp3"
  })
}

#---------------------------------------------------------------
# Kube Prometheus Namespace
#---------------------------------------------------------------
resource "kubernetes_namespace" "kube_prometheus_stack_namespace" {
  count = var.enable_kube_prometheus_stack ? 1 : 0
  metadata {
    name = "kube-prometheus-stack"
  }
}
#---------------------------------------------------------------
# Grafana Admin Password
#---------------------------------------------------------------
resource "random_password" "grafana" {
  count   = var.enable_kube_prometheus_stack ? 1 : 0
  length  = 16
  special = true
}

#---------------------------------------------------------------
# Kubernetes Secret for Grafana Admin
#---------------------------------------------------------------
resource "kubernetes_secret" "grafana_admin" {
  count = var.enable_kube_prometheus_stack ? 1 : 0
  metadata {
    name      = "grafana-admin-secret"
    namespace = "kube-prometheus-stack"
  }

  data = {
    admin-user     = "admin"
    admin-password = random_password.grafana[0].result
  }

  depends_on = [
    kubectl_manifest.kube_prometheus_stack,
    kubernetes_namespace.kube_prometheus_stack_namespace
  ]
}

#---------------------------------------------------------------
# Kube Prometheus Stack Application
#---------------------------------------------------------------
resource "kubectl_manifest" "kube_prometheus_stack" {
  count = var.enable_kube_prometheus_stack ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/kube-prometheus-stack.yaml", {
    user_values_yaml = indent(8, local.kube_prometheus_values)
  })
  wait = true
  depends_on = [
    helm_release.argocd,
    module.amp_ingest_pod_identity
  ]
}

#---------------------------------------------------------------
# Grafana Admin credentials resources
# Login to AWS secrets manager with the same role as Terraform to extract the Grafana admin password with the secret name as "grafana"
#---------------------------------------------------------------
# data "aws_secretsmanager_secret_version" "admin_password_version" {
#   count      = var.enable_kube_prometheus_stack ? 1 : 0
#   secret_id  = aws_secretsmanager_secret.grafana[count.index].id
#   depends_on = [aws_secretsmanager_secret_version.grafana]
# }

# resource "random_password" "grafana" {
#   count            = var.enable_kube_prometheus_stack ? 1 : 0
#   length           = 16
#   special          = true
#   override_special = "@_"
# }

#tfsec:ignore:aws-ssm-secret-use-customer-key
# resource "aws_secretsmanager_secret" "grafana" {
#   count                   = var.enable_kube_prometheus_stack ? 1 : 0
#   name_prefix             = "${local.name}-oss-grafana"
#   recovery_window_in_days = 0 # Set to zero for this example to force delete during Terraform destroy
# }

# resource "aws_secretsmanager_secret_version" "grafana" {
#   count         = var.enable_kube_prometheus_stack ? 1 : 0
#   secret_id     = aws_secretsmanager_secret.grafana[count.index].id
#   secret_string = random_password.grafana[count.index].result
# }
