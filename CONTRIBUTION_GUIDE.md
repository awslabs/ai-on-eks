# Contribution Guide

Thank you! If you've opened this guide, you're probably looking to help contribute to AI on EKS. We really appreciate
you taking the time to help this project. To that end, we've written this guide to help you in your contributions so
that the PR process goes by a lot smoother as well as to remind ourselves of all the things we need to keep in mind when
we make changes to this repository.

## General Contributions

A few things to keep in mind when contributing anything to this repository:

1) Open a Github issue first: before committing code, please open a Github issue describing your proposed changes so we
   can provide guidance and avoid duplicate work.
2) Pre-commit checks: make sure to install pre-commit using `pre-commit install`. This will ensure that terraform is
   linted correctly, whitespace is properly added at the end of the file, and that you are not accidentally committing
   secrets to the repository. You may need to install some tools before precommit will run correctly. Please check
   `.pre-commit-config.yaml` for instructions for each tool.

## Contributing Core Infrastructure

Infrastructure is the basic building block of AI on EKS. AI on EKS is built on the idea of a modular infrastructure: the
infrastructure is separated into discrete components that can be combined to build a platform or illustrate an example (
blueprint).

Infrastructure is separated into 2 levels: Pre Cluster/Managed and Post Cluster

1) Pre Cluster: These are the components that are required for creating a cluster (VPC, subnets, security groups), the
   cluster itself (EKS) and, optionally, other services provided by AWS (databases, observability). The additional EKS
   addons are also part of this level: VPC-CNI, coreDNS, storage controllers, etc.
2) Post Cluster: These are the Kubernetes resources that are deployed after the cluster is running and able to be set up
   for AI. These are things like observability tooling, controllers for AI/ML, self-managed databases (RAG, RDBMS), etc.

All infrastructure is available in `infra/base/terraform`. Pre Cluster infrastructure is created using terraform. You
will find these as separate `.tf` files. If you would like to contribute new Pre Cluster infrastructure, please create a
new terraform resources for it.

All post cluster infrastructure should be deployed by ArgoCD. To create a new ArgoCD deployment for helm, kustomize, or
other, create the ArgoCD Application in `infra/base/terraform/argocd-addons` and add a line in
`infra/base/terraform/argocd_addons.tf` to allow it to be deployed by terraform.

Any infrastructure that is added should be toggleable with the default set to off. The variable should be added to
`infra/base/terraform/variables.tf` where it is overridden by the specific infrastructure that consumes it.
Additionally, `website/docs/infra/ai-ml/index.md` should be updated to reflect the new variable that can be used.

## Contributing Reference Architectures

Reference architectures also live in `infra/`. These are consumers of the available infrastructure modules with a
specific purpose in mind. If you are looking to combine some of the modules together to build a platform or showcase an
example, you can create a new folder in `infra/` named after the architecture you are building. You can then copy the
`install.sh` from any of the other infrastructures into that folder. This will allow creating a copy of the base
terraform into the architecture folder and deploy it. Lastly, create a `terraform` subfolder with a `blueprint.tfvars`
file. This file is used to override the default variables to enable the components you need for your architecture. It is
important to set `name = ARCHITECTURE_NAME`, otherwise it will use the default `ai-stack`. It is also important to note
that `name` is used to name most resources in EKS. It should not be too long, and it should not have any odd characters.
At this point, you can use the variable names in `variables.tf` to override the default `false`, eg:

```terraform
name                    = "my-new-architecture"
enable_kuberay_operator = true
enable_argo_workflows   = true
```

This will deploy everything needed for running EKS for AI along with the Kuberay operator and Argo Workflows.

It is also important to add the infrastructure and its purpose to the website. This is under `website/infra/ai-ml`

A reference follows:

```
ai-on-eks/
├── infra/
│   ├── base/
│   │   └── terraform/
│   │       ├── variables.tf
│   │       ├── argocd-addons/
│   │       └── ...
│   │
│   └── <your-architecture>/           # Your new contribution
│       ├── install.sh                 # Copy from existing architecture
│       └── terraform/
│           └── blueprint.tfvars       # Your custom configuration
├── blueprints/
│   ├── training/
│   ├── inference/
│   └── ...
└── website/
    └── docs/
        └── blueprints/
        └── infra/
```

## Contributing Blueprints

Blueprints are used to highlight specific examples. They run an end-to-end showcase of deploying an infrastructure,
running the example on it, then cleaning up the infrastructure. Where possible, new blueprints should reuse existing
architectures. Contributing blueprints generally requires:

1) New code
2) Documentation

The code should live in `blueprints` under the appropriate category.
Documentation will take the reader through a step-by-step process of how to execute the example. These should go in
`website/blueprints` under the appropriate category.

## Other Contributions

Other contributions are also very much desired. We are looking for benchmarks, best-practices, educational material.
These may require a case-by-case discussion, but will generally fall directly in the `website` folder somewhere.
