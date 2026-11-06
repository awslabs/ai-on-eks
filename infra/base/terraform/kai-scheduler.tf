# KAI Scheduler (GitOps mode)
# Installed from the upstream OCI Helm chart with GitOps overrides
# (kaiConfigDeployer.enabled=false, kaiConfig.render=true) so the kai-config
# Config CR is rendered inline as a tracked resource instead of being applied
# out-of-band by a post-install hook Job.
# See https://github.com/kai-scheduler/KAI-Scheduler/blob/main/docs/gitops/README.md
locals {
  kai_scheduler_values = yamldecode(file("${path.module}/helm-values/kai-scheduler.yaml"))
}

resource "kubectl_manifest" "kai_scheduler_yaml" {
  count = var.enable_kai_scheduler ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/kai-scheduler.yaml", {
    version          = var.kai_scheduler_version
    user_values_yaml = indent(8, yamlencode(local.kai_scheduler_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
