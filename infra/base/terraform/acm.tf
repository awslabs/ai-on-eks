# Use this data source to get the ARN of a certificate in AWS Certificate Manager (ACM)
data "aws_acm_certificate" "issued" {
  count    = var.jupyter_hub_auth_mechanism != "dummy" || var.acm_certificate_domain != "" ? 1 : 0
  domain   = var.acm_certificate_domain
  statuses = ["ISSUED"]
}

data "aws_route53_zone" "hosted_zone_id" {
  count = var.acm_certificate_domain != "" ? 1 : 0
  name  = var.acm_certificate_domain
}
