---
sidebar_position: 0
sidebar_label: Introduction
---

# Introduction

The AI on EKS foundational infrastructure lives in the `infra/base` directory. This directory contains the base
infrastructure and all its modules that allow composing an environment that supports experimentation, AI/ML training,
LLM inference, model tracking, and more.

In the directory is a `variables.tf` which contains all the parameters used to enable or disable desired modules (set to
`false` by default). This enables the ability to deploy a bare environment with Karpenter and GPU and AWS Neuron
`NodePool`s to enable accelerator use and for further customization.

The reference `jark-stack` deploys an environment that facilitates quick AI/ML development by enabling Jupyterhub for
experimentation, the KubeRay operator for training and inference
using [Ray Clusters](https://docs.ray.io/en/latest/cluster/getting-started.html), Argo Workflows for automating
workflows, and storage controllers and volumes. This allows deploying the `notebooks`, `training`, and `inference`
blueprints in the `blueprints` folder.

Other blueprints use the same base infrastructure and selectively enable other components based on the needs of the
blueprint.

## Overview

AI on EKS provides comprehensive infrastructure solutions for deploying AI/ML workloads on Amazon EKS. Choose from pre-configured solutions optimized for training, inference, or general-purpose AI/ML workloads.

### Training Infrastructure

Infrastructure solutions optimized for AI/ML model training workloads:

- **[JARK Stack on EKS](./training/jark.md)** - Complete stack for AI workloads with NVIDIA GPUs, including JupyterHub, Ray, and Kubeflow
- **[JupyterHub on EKS](./training/jupyterhub.md)** - Interactive development environment for data science and ML

### Inference Infrastructure

Infrastructure solutions optimized for AI/ML model inference workloads:

- **[Inference-Ready Cluster](./inference/inference-ready-cluster.md)** - Pre-configured EKS cluster for inference workloads
- **[Nvidia NIM on EKS](../blueprints/inference/framework-guides/GPUs/nvidia-nim-llama3.md)** - Nvidia NIM deployment sample
- **[Nvidia Dynamo on EKS](../blueprints/inference/framework-guides/GPUs/nvidia-dynamo.md)** - Nvidia Dyanmo deployment sample

### Miscellaneous

Additional infrastructure solutions and utilities:

- **[EMR Spark Rapids](./misc/emr-spark-rapids.md)** - GPU-accelerated Apache Spark on Amazon EMR
- **[Troubleshooting](./misc/troubleshooting/troubleshooting.md)** - Common issues and solutions

### Getting Started

1. **Choose Your Use Case**: Select training or inference based on your workload requirements
2. **Deploy Infrastructure**: Follow the deployment guide for your chosen solution
3. **Deploy Workloads**: Use the [Blueprints](../blueprints/index.md) to deploy your AI/ML workloads
4. **Optimize**: Apply [Guidance](../guidance/index.md) best practices

### Architecture Patterns

All infrastructure solutions follow these core principles:

- **Modular Design**: Compose solutions from reusable modules
- **Best Practices**: Built-in security, observability, and scalability
- **Cloud Native**: Leverage Kubernetes and AWS services
- **Validated**: Tested and validated for enterprise workloads

## Resources

Each stack inherits the `base` stack's components. These components include:

- VPC with subnets in 2 availability zones
- EKS cluster with 1 core nodegroup with 2 nodes to run the minimum infrastructure
- Karpenter for autoscaling with CPU, GPU, and AWS Neuron NodePools
- GPU/Neuron device drivers
- GPU/Neuron monitoring agents

## Variables

### Deployment

| Variable Name                            | Description                                         | Default                  |
|------------------------------------------|-----------------------------------------------------|--------------------------|
| `name`                                   | The name of the Kubernetes cluster                  | `ai-stack`               |
| `region`                                 | The region for the cluster                          | us-east-1                |
| `eks_cluster_version`                    | The version of EKS to use                           | 1.32                     |
| `vpc_cidr`                               | The CIDR used for the VPC                           | `10.1.0.0/21`            |
| `secondary_cidr_blocks`                  | Secondary CIDR for the VPC                          | `100.64.0.0/16`          |
| `enable_database_subnets`                | Whether or not to enable the database subnets       | `false`                  |
| `enable_aws_cloudwatch_metrics`          | Enable the AWS Cloudwatch Metrics addon             | `false`                  |
| `bottlerocket_data_disk_snapshot_id`     | Attach a snapshot ID to the deployed nodes          | `""`                     |
| `enable_aws_efs_csi_driver`              | Enable the AWS EFS CSI driver                       | `false`                  |
| `enable_aws_efa_k8s_device_plugin`       | Enable the AWS EFA device plugin                    | `false`                  |
| `enable_aws_fsx_csi_driver`              | Enable the FSx device plugin                        | `false`                  |
| `deploy_fsx_volume`                      | Deploy a simple FSx volume                          | `false`                  |
| `fsx_pvc_namespace`                      | Namespace to provision the FSx PVC                  | `default`                |
| `enable_amazon_prometheus`               | Enable Amazon Managed Prometheus                    | `false`                  |
| `enable_amazon_emr`                      | Set up Amazon EMR                                   | `false`                  |
| `enable_kube_prometheus_stack`           | Enable the Kube Prometheus addon                    | `false`                  |
| `enable_kubecost`                        | Enable Kubecost                                     | `false`                  |
| `enable_ai_ml_observability_stack`       | Enable AI/ML observability addon                    | `false`                  |
| `enable_argo_workflows`                  | Enable Argo Workflow                                | `false`                  |
| `enable_argo_events`                     | Enable Argo Events                                  | `false`                  |
| `enable_argocd`                          | Enable ArgoCD addon                                 | `false`                  |
| `enable_mlflow_tracking`                 | Enable MLFlow Tracking                              | `false`                  |
| `enable_jupyterhub`                      | Enable JupyterHub                                   | `false`                  |
| `enable_volcano`                         | Enable Volcano                                      | `false`                  |
| `enable_kuberay_operator`                | Enable KubeRay                                      | `false`                  |
| `huggingface_token`                      | Hugging Face token to use in environment            | `DUMMY_TOKEN_REPLACE_ME` |
| `enable_rayserve_ha_elastic_cache_redis` | Enable Rayserve high availability using ElastiCache | `false`                  |
| `enable_torchx_etcd`                     | Enable etcd for torchx                              | `false`                  |
| `enable_mpi_operator`                    | Enable the MPI Operator                             | `false`                  |
| `enable_aibrix_stack`                    | Enable the AIBrix stack                             | `false`                  |
| `aibrix_stack_version`                   | AIBrix Stack version                                | `v0.2.1`                 |
| `enable_aws_load_balancer_controller`    | Enable the AWS Load Balancer Controller             | `true`                   |
| `enable_service_mutator_webhook`         | Enable service-mutator webhook for AWS Load Balancer Controller | `false`      |
| `enable_ingress_nginx`                   | Enable ingress-nginx addon                          | `true`                   |
| `enable_cert_manager`                    | Enable Cert Manager                                 | `false`                  |
| `enable_slurm_operator`                  | Enable the Slinky Slurm Operator (with Cert Manager)|  `false`                 |

### JupyterHub

| Variable Name                 | Description                                                                           | Default |
|-------------------------------|---------------------------------------------------------------------------------------|---------|
| `jupyter_hub_auth_mechanism`  | Which authorization mechanism to use for JupyterHub [`dummy` \| `cognito` \| `oauth`] | `dummy` |
| `cognito_custom_domain`       | Cognito domain prefix for Hosted UI authentication endpoints                          | `eks`   |
| `acm_certificate_domain`      | Domain name used for the ACM certificate                                              | `""`    |
| `jupyterhub_domain`           | Domain name for JupyterHub (only used for cognito or oauth)                           | `""`    |
| `oauth_jupyter_client_id`     | oauth clientid for JupyterHub. Only used for oauth                                    | `""`    |
| `oauth_jupyter_client_secret` | oauth client secret. Only used for oauth                                              | `""`    |
| `oauth_username_key`          | oauth field for username (e.g. `preferred_username`). Only needed for oauth           | `""`    |

## Custom Stacks

With the variables above, it's very easy to compose a new environment tailored to your own needs. A `custom` folder is
available in the `infra` folder with a simple `blueprint.tfvars`. By adding the variables above with the appropriate
value, you are able to customize which addons you would like deployed to create an environment to support your
preferences. Once the variables are added, run the `install.sh` at the root of `infra/custom`
