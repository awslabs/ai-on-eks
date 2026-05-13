#---------------------------------------------------------------
# kubernetes-sigs/agent-sandbox controller
#
# The SIG-Apps agent-sandbox project distributes its manifests as
# pre-built release YAMLs (manifest.yaml + extensions.yaml) rather
# than a kustomize-friendly directory tree. ArgoCD's standard Helm /
# directory / kustomize source types don't fit this shape, so we
# use the same pattern as torchx-etcd.tf: fetch the release YAMLs
# via data "http" and apply each document with kubectl_manifest.
#
# When the SIG-Apps project publishes a kustomization manifest or
# Helm chart, this can be migrated to the standard ArgoCD addon
# pattern in argocd_addons.tf.
#---------------------------------------------------------------

data "http" "agent_sandbox_manifest" {
  count = var.enable_agent_sandbox ? 1 : 0
  url   = "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${var.agent_sandbox_version}/manifest.yaml"
}

data "http" "agent_sandbox_extensions" {
  count = var.enable_agent_sandbox ? 1 : 0
  url   = "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${var.agent_sandbox_version}/extensions.yaml"
}

data "kubectl_file_documents" "agent_sandbox_manifest" {
  count   = var.enable_agent_sandbox ? 1 : 0
  content = data.http.agent_sandbox_manifest[0].response_body
}

data "kubectl_file_documents" "agent_sandbox_extensions" {
  count   = var.enable_agent_sandbox ? 1 : 0
  content = data.http.agent_sandbox_extensions[0].response_body
}

resource "kubectl_manifest" "agent_sandbox_manifest" {
  for_each   = var.enable_agent_sandbox ? data.kubectl_file_documents.agent_sandbox_manifest[0].manifests : {}
  yaml_body  = each.value
  depends_on = [module.eks.eks_cluster_id]
}

resource "kubectl_manifest" "agent_sandbox_extensions" {
  for_each   = var.enable_agent_sandbox ? data.kubectl_file_documents.agent_sandbox_extensions[0].manifests : {}
  yaml_body  = each.value
  depends_on = [
    module.eks.eks_cluster_id,
    kubectl_manifest.agent_sandbox_manifest,
  ]
}
