resource "kubectl_manifest" "ai_ml_observability_yaml" {
  count     = var.enable_ai_ml_observability_stack ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/ai-ml-observability.yaml")

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

# NVIDIA K8s DRA Driver
resource "kubectl_manifest" "nvidia_dra_driver" {
  count     = var.enable_nvidia_dra_driver && var.enable_nvidia_gpu_operator ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/nvidia-dra-driver.yaml")

  depends_on = [
    helm_release.argocd
  ]
}

# GPU Operator
resource "kubectl_manifest" "nvidia_gpu_operator" {
  count = var.enable_nvidia_gpu_operator ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-gpu-operator.yaml", {
    service_monitor_enabled = var.enable_ai_ml_observability_stack
  })

  depends_on = [
    helm_release.argocd
  ]
}

# NVIDIA Device Plugin (standalone - GPU scheduling only)
resource "kubectl_manifest" "nvidia_device_plugin" {
  count     = !var.enable_nvidia_gpu_operator && var.enable_nvidia_device_plugin && !var.enable_eks_auto_mode ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-device-plugin.yaml", {})

  depends_on = [
    helm_release.argocd
  ]
}

# DCGM Exporter (standalone - GPU monitoring only)
resource "kubectl_manifest" "nvidia_dcgm_exporter" {
  count = !var.enable_nvidia_gpu_operator && var.enable_nvidia_dcgm_exporter ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-dcgm-exporter.yaml", {
    service_monitor_enabled = var.enable_ai_ml_observability_stack
  })

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
  count     = var.enable_mpi_operator ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/mpi-operator.yaml")

  depends_on = [
    helm_release.argocd,
    kubectl_manifest.cert_manager_yaml
  ]
}

# NVIDIA Dynamo CRDs
resource "kubectl_manifest" "nvidia_dynamo_crds_yaml" {
  count     = var.enable_dynamo_stack ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-dynamo-crds.yaml", { dynamo_version = var.dynamo_stack_version })

  depends_on = [
    helm_release.argocd
  ]
}

# NVIDIA Dynamo Platform
resource "kubectl_manifest" "nvidia_dynamo_platform_yaml" {
  count     = var.enable_dynamo_stack ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/nvidia-dynamo-platform.yaml", { dynamo_version = var.dynamo_stack_version })

  depends_on = [
    helm_release.argocd
  ]
}
