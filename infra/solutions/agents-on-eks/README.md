# Agents on EKS - Agentic Community Reference Deployment

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Plan Your Deployment](#plan-your-deployment)
    - [AWS Services](#aws-services)
- [Security](#security)
- [Prerequisites](#prerequisites)
- [Quick Start Guide](#quick-start-guide)
    - [Deploy the Infrastructure](#deploy-the-infrastructure)
    - [Validate the Deployment](#validate-the-deployment)
- [Accessing Services](#accessing-services)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Cleanup](#cleanup)

## Overview

This infrastructure enables running Agentic AI workloads on Amazon EKS. It deploys an environment that supports
continuously building, deploying, and evaluating AI agents using open source tools in a secure, scalable, and reliable
manner.

Building and operating AI agents at scale requires more than just inference infrastructure. Agents need:

- **Source control and CI/CD** for versioning agent code and configurations
- **Observability** for tracing agent behavior, evaluating performance, and debugging issues
- **Persistent memory** for storing agent memory, embeddings and enabling retrieval-augmented generation (RAG)
- **Tool orchestration** for managing and discovering MCP (Model Context Protocol) servers

This infrastructure brings together these components into a cohesive platform.

## Architecture

This infrastructure creates:

- **Amazon VPC** with public and private subnets across multiple availability zones
- **Amazon EKS Cluster** with managed node groups for critical addons
- **Karpenter** for intelligent node autoscaling based on workload demands
- **GitLab** for source control, container registry, and CI/CD pipelines
- **Langfuse** for agent observability, tracing, and evaluation
- **Milvus** for vector storage and similarity search
- **MCP Gateway Registry** for tool discovery and management

## Components

| Component                                                                         | Version | Purpose                                                   |
|-----------------------------------------------------------------------------------|---------|-----------------------------------------------------------|
| [GitLab](https://about.gitlab.com/)                                               | 9.1.6   | Source control, container registry, and CI/CD pipelines   |
| [Langfuse](https://langfuse.com/)                                                 | 3.124.1 | LLM observability, tracing, prompt management, evaluation |
| [Milvus](https://milvus.io/)                                                      | 2.6.4   | Vector database for embeddings and similarity search      |
| [MCP Gateway Registry](https://github.com/agentic-community/mcp-gateway-registry) | Latest  | Discovery and management of MCP servers                   |
| [ArgoCD](https://argo-cd.readthedocs.io/)                                         | 3.0.6   | GitOps continuous delivery                                |
| [Karpenter](https://karpenter.sh/)                                                | 1.8.1   | Kubernetes node autoscaling                               |
| [External DNS](https://github.com/kubernetes-sigs/external-dns)                   | 0.19.0  | Automatic DNS record management                           |

### GitLab

- **Source Control**: Git repositories for agent code, prompts, and configurations
- **Container Registry**: Store and manage Docker images for agent deployments
- **CI/CD Pipelines**: Automated testing, building, and deployment of agents
- **GitLab Runner**: Kubernetes-native CI/CD job execution with Docker support

### Langfuse

- **Tracing**: Track every LLM call, including inputs, outputs, latency, and costs
- **Prompt Management**: Version and manage prompts with A/B testing
- **Evaluation**: Score and evaluate agent outputs with custom metrics
- **Analytics**: Understand usage patterns and identify optimizations

### Milvus

- **Vector Storage**: Store embeddings for agent memory, RAG and semantic search
- **Similarity Search**: Fast approximate nearest neighbor search
- **Scalability**: Handle billions of vectors with horizontal scaling

### MCP Gateway Registry

- **Tool Discovery**: Register and discover MCP servers
- **Access Control**: Manage which agents can access which tools
- **Monitoring**: Track tool usage and performance metrics

## Plan Your Deployment

### AWS Services

| AWS Service                                                            | Role       | Description                        |
|------------------------------------------------------------------------|------------|------------------------------------|
| [Amazon EKS](https://aws.amazon.com/eks/)                              | Core       | Managed Kubernetes control plane   |
| [Amazon EC2](https://aws.amazon.com/ec2/)                              | Core       | Compute instances for worker nodes |
| [Amazon VPC](https://aws.amazon.com/vpc/)                              | Core       | Isolated network environment       |
| [Amazon Route 53](https://aws.amazon.com/route53/)                     | Core       | DNS management                     |
| [AWS Certificate Manager](https://aws.amazon.com/certificate-manager/) | Core       | TLS certificates                   |
| [Elastic Load Balancing](https://aws.amazon.com/elasticloadbalancing/) | Supporting | Traffic distribution               |
| [Amazon EBS](https://aws.amazon.com/ebs/)                              | Supporting | Persistent storage                 |
| [AWS KMS](https://aws.amazon.com/kms/)                                 | Security   | Encryption key management          |

## Security

### Network Security

- VPC isolation with private subnets for EKS nodes
- Security groups for fine-grained access control
- NAT Gateway for controlled outbound access

### Identity and Access Management

- Pod Identity Policies for accessing AWS services
- Kubernetes RBAC for cluster access control
- Secrets stored in Kubernetes secrets

### Data Protection

- EBS encryption at rest
- TLS for all external endpoints
- KMS integration for encryption keys

## Prerequisites

### Domain and Certificate Setup

GitLab requires a valid TLS certificate, which requires owning a domain.

1. **Create a Route 53 Hosted Zone**

   Follow the [AWS documentation](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/CreatingHostedZone.html).
   For a subdomain, name it `subdomain.domain.tld`.

2. **(Optional) Configure as Subdomain**

   Add NS records from the new hosted zone to your parent domain.

3. **Create an ACM Certificate**

   Follow the [ACM documentation](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html) and
   complete DNS validation.

### Required Tools

- AWS CLI 2.x
- Terraform >= 1.0
- kubectl >= 1.28
- Helm >= 3.0

### Verify Setup

```bash
aws sts get-caller-identity
kubectl version --client
terraform version
helm version
```

## Quick Start Guide

### Deploy the Infrastructure

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/agents-on-eks

# Edit terraform/blueprint.tfvars and set acm_certificate_domain
./install.sh
```

Deployment takes approximately 20 minutes.

### Validate the Deployment

```bash
aws eks update-kubeconfig --name aioeks-agents --region us-west-2
kubectl get pods -A
```

## Accessing Services

### GitLab

Available at `https://gitlab.<your-domain>`.

```bash
# Get root password
kubectl get secret gitlab-gitlab-initial-root-password -n gitlab \
  -o jsonpath='{.data.password}' | base64 -d && echo
```

### Langfuse

```bash
kubectl port-forward svc/langfuse 3000:3000 -n langfuse
# Open http://localhost:3000
```

### Milvus

```bash
# In-cluster: milvus.milvus.svc.cluster.local:19530
kubectl port-forward svc/milvus 19530:19530 -n milvus
```

### MCP Gateway Registry

Available at `https://mcpregistry.<your-domain>`.

### ArgoCD

```bash
kubectl get secret argocd-initial-admin-secret -n argocd \
  -o jsonpath='{.data.password}' | base64 -d && echo
kubectl port-forward svc/argocd-server 8080:443 -n argocd
# Open https://localhost:8080 (user: admin)
```

## Configuration Options

| Variable                      | Description                               | Default         |
|-------------------------------|-------------------------------------------|-----------------|
| `name`                        | Cluster naming prefix                     | `aioeks-agents` |
| `region`                      | AWS region                                | `us-west-2`     |
| `eks_cluster_version`         | EKS version                               | `1.34`          |
| `acm_certificate_domain`      | Domain for TLS                            | `""` (required) |
| `allowed_inbound_cidrs`       | CIDR ranges allowed through load balancer | `0.0.0.0/0`     |
| `enable_langfuse`             | Deploy Langfuse                           | `true`          |
| `enable_gitlab`               | Deploy GitLab                             | `true`          |
| `enable_milvus`               | Deploy Milvus                             | `true`          |
| `enable_mcp_gateway_registry` | Deploy MCP Gateway                        | `true`          |
| `enable_external_dns`         | Enable DNS management                     | `true`          |
| `max_user_namespaces`         | For Docker builds                         | `16384`         |

### Restricting Inbound Access

The `allowed_inbound_cidrs` variable controls which IP ranges can access services through the load balancer. By default,
it allows all traffic (`0.0.0.0/0`).

**For production deployments**, restrict this to your organization's IP ranges:

```hcl
# Example: Allow only your corporate network and VPN
allowed_inbound_cidrs = "10.0.0.0/8,192.168.1.0/24,203.0.113.50/32"
```

**Important**: If using GitLab CI/CD, ensure the CIDR includes:

- Your developer IP addresses (for UI access)
- The GitLab Runner node IPs (for pipeline execution)
- Any external services that need to call your agents

**Note**: The GitLab Runner IP may not be known ahead of time. It is possible to update the ArgoCD application after
deployment or rerun installation after it is added.

## Troubleshooting

### GitLab Runner Issues

```bash
kubectl logs -n gitlab -l app=gitlab-gitlab-runner
```

### Langfuse Database Issues

```bash
kubectl get pods -n langfuse -l app.kubernetes.io/name=postgresql
kubectl logs -n langfuse -l app.kubernetes.io/name=langfuse
```

### DNS Records Not Created

```bash
kubectl logs -n external-dns -l app.kubernetes.io/name=external-dns
```

### Certificate Issues

```bash
aws acm list-certificates --region us-west-2
```

## Cleanup

```bash
cd terraform/_LOCAL
./cleanup.sh
```

**Warning**: This deletes all data including GitLab repos, Langfuse traces, and Milvus vectors.
