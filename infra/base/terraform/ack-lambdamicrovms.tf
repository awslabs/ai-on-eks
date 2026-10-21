# ACK service controller for AWS Lambda MicroVMs (Microvm / MicrovmImage
# CRDs + reconciler). Includes the IRSA role the controller runs under.

data "aws_iam_policy_document" "ack_lambdamicrovms_trust" {
  count = var.enable_ack_lambdamicrovms ? 1 : 0
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:ack-system:ack-lambdamicrovms-controller"]
    }
    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ack_lambdamicrovms" {
  count              = var.enable_ack_lambdamicrovms ? 1 : 0
  name               = "${local.name}-ack-lambdamicrovms"
  assume_role_policy = data.aws_iam_policy_document.ack_lambdamicrovms_trust[0].json
}

resource "aws_iam_role_policy" "ack_lambdamicrovms" {
  count = var.enable_ack_lambdamicrovms ? 1 : 0
  name  = "ack-lambdamicrovms-recommended"
  role  = aws_iam_role.ack_lambdamicrovms[0].id
  # Based on the controller's recommended policy, adjusted from live
  # validation (Sep 2026, us-west-2):
  # - iam:PassRole carries no iam:PassedToService condition. The
  #   lambda-microvms APIs deny the call when the condition is set to
  #   lambda.amazonaws.com. Re-scope once the service documents its
  #   PassedToService value.
  # - lambda:PassNetworkConnector is required: image builds attach an
  #   INTERNET_EGRESS connector to pull the Dockerfile base image.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:CreateMicrovmImage",
          "lambda:UpdateMicrovmImage",
          "lambda:DeleteMicrovmImage",
          "lambda:GetMicrovmImage",
          "lambda:GetMicrovmImageVersion",
          "lambda:ListMicrovmImages",
          "lambda:RunMicrovm",
          "lambda:GetMicrovm",
          "lambda:TerminateMicrovm",
          "lambda:ListMicrovms",
          "lambda:TagResource",
          "lambda:UntagResource",
          "lambda:ListTags"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "lambda:PassNetworkConnector"
        Resource = "arn:aws:lambda:*:*:network-connector:*"
      }
    ]
  })
}

locals {
  ack_lambdamicrovms_values = var.enable_ack_lambdamicrovms ? templatefile("${path.module}/helm-values/ack-lambdamicrovms.yaml", {
    region   = var.region
    role_arn = aws_iam_role.ack_lambdamicrovms[0].arn
  }) : ""
}

resource "kubectl_manifest" "ack_lambdamicrovms_yaml" {
  count = var.enable_ack_lambdamicrovms ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/ack-lambdamicrovms.yaml", {
    version          = var.ack_lambdamicrovms_version
    user_values_yaml = indent(8, local.ack_lambdamicrovms_values)
  })
  depends_on = [
    helm_release.argocd
  ]
}
