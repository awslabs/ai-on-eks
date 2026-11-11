---
name: argocd-addon
description: Create a new ArgoCD addon in this project. Use when adding a Kubernetes application deployed via ArgoCD and managed through Terraform, including the ArgoCD Application manifest, Terraform resources, and variables.
---

# Create an ArgoCD Addon

An addon is a Kubernetes application deployed via ArgoCD and managed through Terraform.

## Workflow

> **IMPORTANT**: This skill is the single source of truth for addon structure. Do NOT read existing repo files (`argocd_addons.tf`, `variables.tf`, other addon YAMLs) for reference — everything you need is defined below. The only external research needed is the addon's upstream Helm chart URL and latest version number. Go straight from this skill's templates to creating files.

### Pattern Selection (REQUIRED)

If the user has NOT specified which pattern to use (A, B, C, or D), you MUST ask them to choose one before proceeding. Present this summary:

> Which addon pattern should I use?
>
> - **Pattern A – Simple Static**: No dynamic values. The YAML is applied as-is with no Terraform templating. Best for addons with a fixed version and no customization needed.
> - **Pattern B – Version Templating**: The chart version (and optionally a few other scalars) are injected via Terraform `templatefile`. Best when you want the version configurable but don't need custom helm values.
> - **Pattern C – Helm Values File**: A separate `helm-values/<addon>.yaml` provides rich, customizable helm values passed into the ArgoCD Application. Best for addons that need user-tunable configuration.
> - **Pattern D – Git-Based Source**: Deployed from a git repo using kustomize or plain manifests instead of a Helm chart. Best for in-house or non-Helm addons.

Do NOT proceed with file creation until the user has selected a pattern.

## Project Layout

All addon files live under `infra/base/terraform/`:

```
infra/base/terraform/
├── argocd_addons.tf                          # Terraform resources for Pattern A / B addons
├── argocd-addons/                            # ArgoCD Application YAML manifests
│   ├── <addon-name>.yaml
│   └── ...
├── helm-values/                              # Helm values files (Pattern C addons)
│   ├── <addon-name>.yaml
│   └── ...
├── variables.tf                              # Terraform variables (enable flags, versions, etc.)
└── <addon-name>.tf                           # Dedicated .tf file (required for every Pattern C addon)
```

## Addon Patterns

There are three tiers of complexity. Choose the one that fits.

---

### Pattern A: Simple Static Addon (no templating)

Use when the addon YAML has no dynamic values at all.

**1. ArgoCD Application manifest** — `argocd-addons/<addon-name>.yaml`

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: <addon-name>
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: <helm-repo-url>
    chart: <chart-name>
    targetRevision: <pinned-version>
  destination:
    server: https://kubernetes.default.svc
    namespace: <target-namespace>
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
    syncOptions:
      - CreateNamespace=true
      - ServerSideApply=true
```

**2. Terraform resource** — append to `argocd_addons.tf`

```hcl
resource "kubectl_manifest" "<addon_name_underscored>" {
  count     = var.enable_<addon_name_underscored> ? 1 : 0
  yaml_body = file("${path.module}/argocd-addons/<addon-name>.yaml")

  depends_on = [
    helm_release.argocd
  ]
}
```

**3. Terraform variables** — append to `variables.tf`

```hcl
variable "enable_<addon_name_underscored>" {
  description = "Enable <Addon Display Name> addon"
  type        = bool
  default     = false
}
```



---

### Pattern B: Addon with Version Templating

Use when the version (and possibly a few other scalar values) need to be configurable via Terraform variables.

**1. ArgoCD Application manifest** — `argocd-addons/<addon-name>.yaml`

Same as Pattern A but use `${version}` in `targetRevision`. If the chart/repo tags require a `v` prefix, add it in the YAML as `v${version}`. The Terraform variable default must always store the bare version number without any `v` prefix.

```yaml
    targetRevision: v${version}   # add "v" prefix here if the chart requires it
```

**2. Terraform resource** — append to `argocd_addons.tf`

```hcl
resource "kubectl_manifest" "<addon_name_underscored>" {
  count = var.enable_<addon_name_underscored> ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/<addon-name>.yaml", {
    version = var.<addon_name_underscored>_version
  })

  depends_on = [
    helm_release.argocd
  ]
}
```

**3. Terraform variables** — append to `variables.tf`

```hcl
variable "enable_<addon_name_underscored>" {
  description = "Enable <Addon Display Name> addon"
  type        = bool
  default     = false
}

variable "<addon_name_underscored>_version" {
  description = "<Addon Display Name> chart version"
  type        = string
  default     = "<default-version>"
}
```



---

### Pattern C: Addon with Helm Values File

Use when the addon needs rich helm values that users may want to customize. This is the most flexible pattern.

**1. Helm values file** — `helm-values/<addon-name>.yaml`

Plain YAML with the default helm values. Can optionally use Terraform `templatefile` variables (e.g. `${prometheus_endpoint}`).

```yaml
someKey:
  nestedKey: value
```

**2. ArgoCD Application manifest** — `argocd-addons/<addon-name>.yaml`

Include `${user_values_yaml}` under `helm.values`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: <addon-name>
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: <helm-repo-url>
    chart: <chart-name>
    targetRevision: ${version}
    helm:
      values: |
        ${user_values_yaml}
  destination:
    server: https://kubernetes.default.svc
    namespace: <target-namespace>
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
    syncOptions:
      - CreateNamespace=true
      - ServerSideApply=true
```

**3. Terraform resource** — create a dedicated `<addon-name>.tf` file (do NOT append to `argocd_addons.tf`)

Pattern C addons always get their own `<addon-name>.tf` file (kebab-case, matching the YAML file name). Put the `locals` block and the `kubectl_manifest` resource in that file. This keeps the helm-values wiring for each customizable addon self-contained and easy to find.

If the helm-values file has NO template variables:

```hcl
locals {
  <addon_name_underscored>_values = yamldecode(file("${path.module}/helm-values/<addon-name>.yaml"))
}

resource "kubectl_manifest" "<addon_name_underscored>" {
  count = var.enable_<addon_name_underscored> ? 1 : 0
  yaml_body = templatefile("${path.module}/argocd-addons/<addon-name>.yaml", {
    version          = var.<addon_name_underscored>_version
    user_values_yaml = indent(8, yamlencode(local.<addon_name_underscored>_values))
  })

  depends_on = [
    helm_release.argocd
  ]
}
```

If the helm-values file itself uses template variables (like dynamo does):

```hcl
locals {
  <addon_name_underscored>_values = yamldecode(templatefile("${path.module}/helm-values/<addon-name>.yaml", {
    some_variable = var.some_variable
  }))
}
```

**4. Terraform variables** — append to `variables.tf`

```hcl
variable "enable_<addon_name_underscored>" {
  description = "Enable <Addon Display Name> addon"
  type        = bool
  default     = false
}

variable "<addon_name_underscored>_version" {
  description = "<Addon Display Name> chart version"
  type        = string
  default     = "<default-version>"
}
```



---

### Pattern D: Git-Based Source (no Helm chart)

Use when the addon is deployed from a git repository using kustomize or plain manifests instead of a Helm chart.

```yaml
spec:
  source:
    repoURL: https://github.com/<org>/<repo>.git
    targetRevision: ${version}
    path: <path/to/manifests>
```

---

## Naming Conventions

- YAML file name: kebab-case (`myaddon-example.yaml`)
- Terraform resource name: snake_case (`myaddon_example`)
- Terraform variable prefix: `enable_myaddon_example` and `myaddon_example_version`
- ArgoCD Application `metadata.name`: kebab-case, matching the YAML file name
- Kubernetes namespace: typically `<addon-name>-system` or `<addon-name>`

## Version Convention

- Terraform variable defaults must always store the bare version number without any prefix (e.g. `"0.12.6"`, `"1.14.1"`).
- If the chart or git repo requires a `v` prefix on the tag, add it in the ArgoCD Application YAML as `v${version}`. Never store the `v` in the Terraform variable.

## Dependencies

- Every addon resource MUST have `depends_on = [helm_release.argocd]`
- If the addon depends on another addon (e.g. slurm depends on cert-manager), add that to `depends_on` as well
- If the addon has CRDs that must be installed first, create a separate CRD manifest and resource with its own `kubectl_manifest` as argocd addon, and make the main addon depend on it (see kuberay-operator pattern)

## Enabling an Addon

Users enable addons in their blueprint tfvars file (e.g. `blueprint.tfvars`):

```hcl
enable_myaddon_example = true
myaddon_example_version = "1.11.0"
```

## Optional Sections in the ArgoCD Application

- `finalizers`: Add `resources-finalizer.argocd.argoproj.io` if ArgoCD should clean up resources on Application deletion
- `ignoreDifferences`: Use when controllers mutate resources after apply (e.g. webhook caBundle, ClusterRole rules)
- `annotations`: Use `argocd.argoproj.io/compare-options: ServerSideDiff=true` for large or complex resources
- `helm.skipCrds: true`: When CRDs are managed separately or by the operator itself
