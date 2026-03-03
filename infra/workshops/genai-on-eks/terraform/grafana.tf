################################################################################
# Observability Stack - Grafana Dashboards
# Grafana Operator is deployed in base terraform (enable_grafana_operator=true)
# Grafana credentials are managed by base terraform (grafana-admin-secret)
# This file manages workshop-specific dashboards
# Note: Namespace is configured via kube_prometheus_stack_namespace variable
################################################################################

locals {
  amp_datasource_name = "Amazon-Managed-Prometheus"

  # Transform dashboards to use AMP datasource when enabled
  dashboards = {
    vllm = var.enable_amazon_prometheus ? replace(replace(
      file("${path.module}/grafana-dashboards/vllm-dashboard.json"),
      "\"type\": \"prometheus\"", "\"type\": \"grafana-amazonprometheus-datasource\""),
      "\"uid\": \"prometheus\"", "\"uid\": \"${local.amp_datasource_name}\""
    ) : file("${path.module}/grafana-dashboards/vllm-dashboard.json")

    ray_default = var.enable_amazon_prometheus ? replace(replace(
      file("${path.module}/grafana-dashboards/ray-default-grafana-dashboard.json"),
      "\"type\": \"prometheus\"", "\"type\": \"grafana-amazonprometheus-datasource\""),
      "\"uid\": \"prometheus\"", "\"uid\": \"${local.amp_datasource_name}\""
    ) : file("${path.module}/grafana-dashboards/ray-default-grafana-dashboard.json")

    ray_serve = var.enable_amazon_prometheus ? replace(replace(
      file("${path.module}/grafana-dashboards/ray-serve-grafana-dashboard.json"),
      "\"type\": \"prometheus\"", "\"type\": \"grafana-amazonprometheus-datasource\""),
      "\"uid\": \"prometheus\"", "\"uid\": \"${local.amp_datasource_name}\""
    ) : file("${path.module}/grafana-dashboards/ray-serve-grafana-dashboard.json")

    ray_serve_deployment = var.enable_amazon_prometheus ? replace(replace(
      file("${path.module}/grafana-dashboards/ray-serve-deployment-grafana-dashboard.json"),
      "\"type\": \"prometheus\"", "\"type\": \"grafana-amazonprometheus-datasource\""),
      "\"uid\": \"prometheus\"", "\"uid\": \"${local.amp_datasource_name}\""
    ) : file("${path.module}/grafana-dashboards/ray-serve-deployment-grafana-dashboard.json")

    dcgm = var.enable_amazon_prometheus ? replace(replace(
      file("${path.module}/grafana-dashboards/dcgm-grafana-dashboard.json"),
      "\"type\": \"prometheus\"", "\"type\": \"grafana-amazonprometheus-datasource\""),
      "\"uid\": \"prometheus\"", "\"uid\": \"${local.amp_datasource_name}\""
    ) : file("${path.module}/grafana-dashboards/dcgm-grafana-dashboard.json")
  }
}

resource "kubectl_manifest" "external_grafana" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: grafana.integreatly.org/v1beta1
    kind: Grafana
    metadata:
      name: external-grafana
      namespace: ${var.kube_prometheus_stack_namespace}
      labels:
        dashboards: external-grafana
    spec:
      external:
        url: http://kube-prometheus-stack-grafana.${var.kube_prometheus_stack_namespace}.svc.cluster.local:${var.grafana_service_port}
        adminUser:
          name: grafana-admin-secret
          key: admin-user
        adminPassword:
          name: grafana-admin-secret
          key: admin-password
  YAML
}

resource "kubectl_manifest" "vllm_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: vllm-grafana-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      vllm-dashboard.json: ${jsonencode(local.dashboards.vllm)}
  YAML
}

resource "kubectl_manifest" "ray_default_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: ray-grafana-default-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      ray-default-grafana-dashboard.json: ${jsonencode(local.dashboards.ray_default)}
  YAML
}

resource "kubectl_manifest" "ray_serve_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: ray-grafana-serve-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      ray-serve-grafana-dashboard.json: ${jsonencode(local.dashboards.ray_serve)}
  YAML
}

resource "kubectl_manifest" "ray_serve_deployment_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: ray-grafana-serve-deployment-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      ray-serve-deployment-grafana-dashboard.json: ${jsonencode(local.dashboards.ray_serve_deployment)}
  YAML
}

resource "kubectl_manifest" "dcgm_grafana_dashboard_config" {
  depends_on = [module.eks]

  yaml_body = <<-YAML
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: dcgm-dashboard-config
      namespace: ${var.kube_prometheus_stack_namespace}
    data:
      dcgm-grafana-dashboard.json: ${jsonencode(local.dashboards.dcgm)}
  YAML
}

################################################################################
# GrafanaDashboard CRDs - Register dashboards with Grafana Operator
################################################################################

resource "kubectl_manifest" "vllm_grafana_dashboard" {
  depends_on = [
    kubectl_manifest.external_grafana,
    kubectl_manifest.vllm_grafana_dashboard_config
  ]

  yaml_body = <<-YAML
    apiVersion: grafana.integreatly.org/v1beta1
    kind: GrafanaDashboard
    metadata:
      name: vllm-dashboard
      namespace: ${var.kube_prometheus_stack_namespace}
    spec:
      instanceSelector:
        matchLabels:
          dashboards: external-grafana
      configMapRef:
        name: vllm-grafana-dashboard-config
        key: vllm-dashboard.json
  YAML
}

resource "kubectl_manifest" "ray_default_grafana_dashboard" {
  depends_on = [
    kubectl_manifest.external_grafana,
    kubectl_manifest.ray_default_grafana_dashboard_config
  ]

  yaml_body = <<-YAML
    apiVersion: grafana.integreatly.org/v1beta1
    kind: GrafanaDashboard
    metadata:
      name: ray-default-dashboard
      namespace: ${var.kube_prometheus_stack_namespace}
    spec:
      instanceSelector:
        matchLabels:
          dashboards: external-grafana
      configMapRef:
        name: ray-grafana-default-dashboard-config
        key: ray-default-grafana-dashboard.json
  YAML
}

resource "kubectl_manifest" "ray_serve_grafana_dashboard" {
  depends_on = [
    kubectl_manifest.external_grafana,
    kubectl_manifest.ray_serve_grafana_dashboard_config
  ]

  yaml_body = <<-YAML
    apiVersion: grafana.integreatly.org/v1beta1
    kind: GrafanaDashboard
    metadata:
      name: ray-serve-dashboard
      namespace: ${var.kube_prometheus_stack_namespace}
    spec:
      instanceSelector:
        matchLabels:
          dashboards: external-grafana
      configMapRef:
        name: ray-grafana-serve-dashboard-config
        key: ray-serve-grafana-dashboard.json
  YAML
}

resource "kubectl_manifest" "ray_serve_deployment_grafana_dashboard" {
  depends_on = [
    kubectl_manifest.external_grafana,
    kubectl_manifest.ray_serve_deployment_grafana_dashboard_config
  ]

  yaml_body = <<-YAML
    apiVersion: grafana.integreatly.org/v1beta1
    kind: GrafanaDashboard
    metadata:
      name: ray-serve-deployment-dashboard
      namespace: ${var.kube_prometheus_stack_namespace}
    spec:
      instanceSelector:
        matchLabels:
          dashboards: external-grafana
      configMapRef:
        name: ray-grafana-serve-deployment-dashboard-config
        key: ray-serve-deployment-grafana-dashboard.json
  YAML
}

resource "kubectl_manifest" "dcgm_grafana_dashboard" {
  depends_on = [
    kubectl_manifest.external_grafana,
    kubectl_manifest.dcgm_grafana_dashboard_config
  ]

  yaml_body = <<-YAML
    apiVersion: grafana.integreatly.org/v1beta1
    kind: GrafanaDashboard
    metadata:
      name: dcgm-dashboard
      namespace: ${var.kube_prometheus_stack_namespace}
    spec:
      instanceSelector:
        matchLabels:
          dashboards: external-grafana
      configMapRef:
        name: dcgm-dashboard-config
        key: dcgm-grafana-dashboard.json
  YAML
}
