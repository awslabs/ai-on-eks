resource "kubectl_manifest" "ai_ml_observability_yaml" {
  count = var.enable_ai_ml_observability_stack ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/ai-ml-observability.yaml", {
    observability_mcp_enabled = var.observability_mcp_enabled
  })

  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "kuberay_operator_crds" {
  count     = var.enable_kuberay_operator ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/kuberay-operator-crds.yaml", { kuberay_version = var.kuberay_operator_version })

  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "kuberay_operator" {
  count     = var.enable_kuberay_operator ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/kuberay-operator.yaml", { kuberay_version = var.kuberay_operator_version })

  depends_on = [
    helm_release.argocd,
    kubectl_manifest.kuberay_operator_crds
  ]
}

resource "kubectl_manifest" "aibrix_dependency_yaml" {
  count     = var.enable_aibrix_stack ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/aibrix-dependency.yaml", { aibrix_version = var.aibrix_stack_version })

  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "aibrix_core_yaml" {
  count     = var.enable_aibrix_stack ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/aibrix-core.yaml", { aibrix_version = var.aibrix_stack_version })

  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "envoy_ai_gateway_crds_yaml" {
  count     = var.enable_envoy_ai_gateway_crds ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/envoy-ai-gateway-crds.yaml")
  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "envoy_ai_gateway_yaml" {
  count     = var.enable_envoy_ai_gateway ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/envoy-ai-gateway.yaml")
  depends_on = [
    helm_release.argocd,
    kubectl_manifest.envoy_ai_gateway_crds_yaml
  ]
}

resource "kubectl_manifest" "redis_yaml" {
  count     = var.enable_redis ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/redis.yaml")
  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "envoy_gateway_yaml" {
  count     = var.enable_envoy_gateway ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/envoy-gateway.yaml")
  depends_on = [
    helm_release.argocd,
    kubectl_manifest.redis_yaml
  ]
}

resource "kubectl_manifest" "lws_yaml" {
  count     = var.enable_leader_worker_set ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/leader-worker-set.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

resource "kubectl_manifest" "nvidia_nim_yaml" {
  count     = var.enable_nvidia_nim_stack ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/nvidia-nim-operator.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

# Cert Manager
resource "kubectl_manifest" "cert_manager_yaml" {
  count     = var.enable_cert_manager || var.enable_slurm_operator ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/cert-manager.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

# MariaDB Operator
resource "kubectl_manifest" "mariadb_operator_yaml" {
  count     = var.enable_mariadb_operator ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/mariadb-operator.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

# Slinky Slurm Operator
resource "kubectl_manifest" "slurm_operator_yaml" {
  count     = var.enable_slurm_operator ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/slurm-operator.yaml")

  depends_on = [
    helm_release.argocd,
    kubectl_manifest.cert_manager_yaml
  ]
}

# MPI Operator
resource "kubectl_manifest" "mpi_operator" {
  count = var.enable_mpi_operator ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/mpi-operator.yaml", {
    version = var.mpi_operator_version
  })

  depends_on = [
    helm_release.argocd,
    kubectl_manifest.cert_manager_yaml
  ]
}

# Langfuse
resource "kubectl_manifest" "langfuse_yaml" {
  count     = var.enable_langfuse ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/observability/langfuse/langfuse.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

# Langfuse Secret
# TODO: Move this

resource "random_bytes" "langfuse_secret" {
  count  = var.enable_langfuse ? 8 : 0
  length = 32
}

resource "kubectl_manifest" "langfuse_secret_yaml" {
  count = var.enable_langfuse ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/observability/langfuse/langfuse-secret.yaml", {
    salt                = random_bytes.langfuse_secret[0].hex
    encryption-key      = random_bytes.langfuse_secret[1].hex
    nextauth-secret     = random_bytes.langfuse_secret[2].hex
    postgresql-password = random_bytes.langfuse_secret[3].hex
    clickhouse-password = random_bytes.langfuse_secret[4].hex
    redis-password      = random_bytes.langfuse_secret[5].hex
    s3-user             = random_bytes.langfuse_secret[6].hex
    s3-password         = random_bytes.langfuse_secret[7].hex
  })

  depends_on = [
    kubectl_manifest.langfuse_yaml
  ]
}

# Gitlab
resource "kubectl_manifest" "gitlab_yaml" {
  count = var.enable_gitlab ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/devops/gitlab/gitlab.yaml", {
    proxy-real-ip-cidr    = local.vpc_cidr
    acm_certificate_arn   = data.aws_acm_certificate.issued[0].arn
    domain                = var.acm_certificate_domain
    allowed_inbound_cidrs = var.allowed_inbound_cidrs
  })

  depends_on = [
    helm_release.argocd
  ]
}

# Milvus
resource "kubectl_manifest" "milvus_yaml" {
  count = var.enable_milvus ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/vector-databases/milvus/milvus.yaml", {
  })

  depends_on = [
    helm_release.argocd
  ]
}


# Selenium Grid
resource "kubectl_manifest" "selenium_grid_yaml" {
  count     = var.enable_selenium_grid ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/selenium-grid.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

# Jupyter Enterprise Gateway
resource "kubectl_manifest" "jupyter_enterprise_gateway_yaml" {
  count     = var.enable_jupyter_enterprise_gateway ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/jupyter-enterprise-gateway.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

# MCP Gateway Registry
resource "kubectl_manifest" "mcp_gateway_registry_yaml" {
  count = var.enable_mcp_gateway_registry ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/mcp-gateway-registry.yaml", {
    domain                = var.acm_certificate_domain
    allowed_inbound_cidrs = var.allowed_inbound_cidrs
  })

  depends_on = [
    helm_release.argocd
  ]
}

# Gateway API CRDs
resource "kubectl_manifest" "gateway_api_crds_yaml" {
  count = var.enable_gateway_api_crds ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/gateway-api-crds.yaml", {
    version = var.gateway_api_crds_version
  })

  depends_on = [
    helm_release.argocd
  ]
}

# Gateway API Inference Extension CRDs
resource "kubectl_manifest" "gateway_api_inference_crds_yaml" {
  count = var.enable_gateway_api_inference_crds ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/gateway-api-inference-crds.yaml", {
    version = var.gateway_api_inference_crds_version
  })

  depends_on = [
    helm_release.argocd
  ]
}
