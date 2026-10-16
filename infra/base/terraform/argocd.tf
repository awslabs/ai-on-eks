
#---------------------------------------------------------------
# ArgoCD Installation via Terraform
#---------------------------------------------------------------
resource "helm_release" "argocd" {
  name             = "argocd"
  repository       = "https://argoproj.github.io/argo-helm"
  chart            = "argo-cd"
  version          = "8.6.4"
  namespace        = "argocd"
  create_namespace = true

  values = [
    <<-EOT
    configs:
      cm:
        kustomize.buildOptions: --enable-helm
        application.resourceTrackingMethod: annotation

    dex:
      enabled: false

    notifications:
      enabled: false

    EOT
  ]

  depends_on = [
    module.eks.cluster_id,
    kubectl_manifest.nodepools_manifests,
    aws_eks_addon.aws_ebs_csi_driver
  ]
}
