resource "kubectl_manifest" "external_dns" {
  count = var.enable_external_dns && var.acm_certificate_domain != "" ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/external-dns.yaml", {
    user_values_yaml = ""
  })

  depends_on = [
    helm_release.argocd,
    module.external_dns_pod_identity
  ]
}



module "external_dns_pod_identity" {
  count  = var.enable_external_dns && var.acm_certificate_domain != "" ? 1 : 0
  source = "terraform-aws-modules/eks-pod-identity/aws"

  name = "external-dns"

  attach_external_dns_policy    = true
  external_dns_hosted_zone_arns = ["arn:aws:route53:::hostedzone/${data.aws_route53_zone.hosted_zone_id[0].zone_id}"]
  associations = {
    external_dns = {
      cluster_name    = module.eks.cluster_name
      namespace       = "external-dns"
      service_account = "external-dns"
    }
  }
  tags = local.tags
}
