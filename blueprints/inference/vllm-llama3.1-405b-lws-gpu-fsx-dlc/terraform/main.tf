# Terraform deploy path for the multi-node Llama-3.1-405B vLLM (AWS DLC) + FSx-over-EFA blueprint.
# Alternative to `kubectl apply` (see ../README.md). Installs the LeaderWorkerSet controller — the
# ai-on-eks base does NOT ship it — and renders + applies the serving group. The NVIDIA and EFA
# device plugins ARE provided by the base, so they are intentionally NOT installed here.
#
# DLC variant: the image base is the AWS vLLM Deep Learning Container (ships aws-ofi-nccl), so
# inter-node NCCL rides EFA. Extends AWS's single-node vLLM DLC EKS example to multi-node.
#
# Prereqs: base deployed with the FSx-EFA options on (deploy_fsx_volume, fsx_efa_enabled, FSx CSI,
# an EFA GPU node pool); model weights staged on FSx; your DLC image built + pushed (see ../README.md).

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws        = { source = "hashicorp/aws", version = ">= 5.60" }
    kubernetes = { source = "hashicorp/kubernetes", version = ">= 2.30" }
    helm       = { source = "hashicorp/helm", version = "~> 2.13" }
    local      = { source = "hashicorp/local", version = ">= 2.4" }
    null       = { source = "hashicorp/null", version = ">= 3.2" }
  }
}

variable "region" {
  description = "AWS region of the target EKS cluster."
  type        = string
}
variable "cluster_name" {
  description = "Name of the target EKS cluster."
  type        = string
}

# Custom DLC image: AWS vLLM DLC + Ray + Run:AI streamer (see ../README.md). Required.
variable "vllm_image" {
  description = "Serving container image (ECR URI): AWS vLLM DLC + Ray + Run:AI Model Streamer. Required."
  type        = string
}

variable "mount_path" {
  description = "In-container mount path for the FSx for Lustre volume."
  type        = string
  default     = "/mnt/fsx"
}
variable "model_path" {
  description = "Path to the model weights on the mounted FSx volume."
  type        = string
  default     = "/mnt/fsx/models/Llama-3.1-405B-Instruct"
}
variable "served_model_name" {
  description = "Model name advertised by the OpenAI-compatible server."
  type        = string
  default     = "llama-3.1-405b"
}
# Static FSx PVC provided by the base (infra/base/terraform/fsx-for-lustre.tf).
variable "pvc_name" {
  description = "Name of the static FSx for Lustre PVC provided by the base."
  type        = string
  default     = "fsx-static-pvc"
}
variable "gpu_instance_type" {
  description = "GPU instance type for the serving nodes."
  type        = string
  default     = "p5.48xlarge"
}
variable "gpus_per_node" {
  description = "Number of GPUs per node."
  type        = number
  default     = 8
}
variable "efa_per_pod" {
  description = "Number of EFA interfaces requested per pod (p5.48xlarge=32, p5en=16)."
  type        = number
  default     = 32 # p5.48xlarge=32 (p5en=16)
}
variable "hugepages_2mi" {
  description = "2Mi hugepages requested per pod for the DLC NCCL-over-EFA path."
  type        = string
  default     = "5120Mi" # DLC/NCCL-over-EFA path requests hugepages
}
variable "pod_memory" {
  description = "Memory request/limit per pod."
  type        = string
  default     = "1600Gi"
}
variable "shm_size" {
  description = "Shared-memory (/dev/shm) size for the pod."
  type        = string
  default     = "512Gi"
}
variable "worker_replicas" {
  description = "LWS worker replicas; serving-group size = replicas + 1 = node count = pipeline-parallel degree."
  type        = number
  default     = 1 # size = replicas + 1 = node count = PP degree
}
variable "tensor_parallel_size" {
  description = "Tensor-parallel degree (intra-node, over NVLink)."
  type        = number
  default     = 8 # intra-node (NVLink)
}
variable "pipeline_parallel_size" {
  description = "Pipeline-parallel degree (inter-node, NCCL over EFA on the DLC image)."
  type        = number
  default     = 2 # inter-node (NCCL over EFA on the DLC image)
}
variable "max_model_len" {
  description = "Maximum model context length in tokens."
  type        = number
  default     = 8192
}
variable "runai_concurrency" {
  description = "Run:AI Model Streamer concurrency for weight loading."
  type        = number
  default     = 32
}
variable "lws_chart_version" {
  description = "LeaderWorkerSet Helm chart version (validated pin; newer exist — see note)."
  type        = string
  default     = "0.9.0" # validated pin; newer LWS exist (up to v1.0.0 as of 2026-10; v1.0.0 has breaking changes) — bump at your discretion
}

provider "aws" { region = var.region }

data "aws_eks_cluster" "this" { name = var.cluster_name }
data "aws_eks_cluster_auth" "this" { name = var.cluster_name }

provider "kubernetes" {
  host                   = data.aws_eks_cluster.this.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.this.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.this.token
}

provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.this.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.this.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.this.token
  }
}

# LeaderWorkerSet controller (official OCI Helm chart). The base does NOT install LWS; it DOES provide
# the NVIDIA + EFA device plugins, so those installs are intentionally omitted here.
resource "helm_release" "lws" {
  name             = "lws"
  repository       = "oci://registry.k8s.io/lws/charts"
  chart            = "lws"
  version          = var.lws_chart_version
  namespace        = "lws-system"
  create_namespace = true
  wait             = true
}

# Render the LWS vLLM serving group from the template.
resource "local_file" "lws_manifest" {
  filename = "${path.module}/.rendered/vllm-405b-lws.yaml"
  content = templatefile("${path.module}/vllm-405b-lws.yaml.tftpl", {
    image             = var.vllm_image
    gpu_instance      = var.gpu_instance_type
    size              = var.worker_replicas + 1
    gpus_per_node     = var.gpus_per_node
    efa_per_pod       = var.efa_per_pod
    hugepages         = var.hugepages_2mi
    pod_memory        = var.pod_memory
    shm_size          = var.shm_size
    total_gpus        = var.gpus_per_node * (var.worker_replicas + 1)
    model_path        = var.model_path
    mount_path        = var.mount_path
    served_model_name = var.served_model_name
    pvc_name          = var.pvc_name
    tp                = var.tensor_parallel_size
    pp                = var.pipeline_parallel_size
    max_model_len     = var.max_model_len
    runai_concurrency = var.runai_concurrency
  })
}

# Apply the rendered manifest. local-exec kubectl apply (not kubernetes_manifest) avoids the
# plan-time CRD problem: the LeaderWorkerSet CRD is created by helm_release.lws in this same apply.
resource "null_resource" "apply_lws" {
  depends_on = [helm_release.lws]
  triggers   = { manifest_sha = local_file.lws_manifest.content_md5 }
  provisioner "local-exec" {
    command = <<-EOT
      aws eks update-kubeconfig --name ${var.cluster_name} --region ${var.region}
      kubectl apply -f ${local_file.lws_manifest.filename}
    EOT
  }
}

output "endpoint" {
  value       = "kubectl exec <pod> -- curl -s http://llm405b-leader.default.svc.cluster.local:8000/v1/models"
  description = "In-cluster OpenAI-compatible endpoint (served by the LWS leader)."
}
