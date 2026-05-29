---
sidebar_label: Agents on EKS
---

# Agents on EKS

The Agents on EKS infrastructure deploys an environment that supports continuously building, deploying, and evaluating AI agents using open source tools in a secure, scalable, and reliable manner.

## Why?

Building and operating AI agents at scale requires more than just inference infrastructure. Agents need:

- **Source control and CI/CD** for versioning agent code and configurations
- **Observability** for tracing agent behavior, evaluating performance, and debugging issues
- **Persistent memory** for storing embeddings and enabling retrieval-augmented generation (RAG)
- **Tool orchestration** for managing and discovering MCP (Model Context Protocol) servers

This infrastructure brings together these components into a cohesive platform, enabling teams to iterate quickly on agent development while maintaining reliability.

## Use Cases

- **Agent Development**: Build and test AI agents with integrated source control and CI/CD pipelines
- **Agent Evaluation**: Use Langfuse to trace agent executions, evaluate outputs, and track performance over time
- **RAG Applications**: Store and retrieve embeddings using Milvus for knowledge-augmented agents
- **MCP Tool Management**: Discover and manage MCP servers through the gateway registry
- **Multi-Agent Systems**: Deploy and orchestrate multiple agents with shared infrastructure

## Architecture

This infrastructure creates:

- **Amazon VPC** with public and private subnets across multiple availability zones
- **Amazon EKS Cluster** with managed node groups for critical addons
- **Karpenter** for intelligent node autoscaling based on workload demands
- **GitLab** for source control, container registry, and CI/CD pipelines
- **Langfuse** for agent observability, tracing, and evaluation
- **Milvus** for vector storage and similarity search
- **MCP Gateway Registry** for tool discovery and management

### Core Components

| Component | Purpose |
|-----------|---------|
| [GitLab](https://about.gitlab.com/) | Source control, container registry, and CI/CD for agent code |
| [Langfuse](https://langfuse.com/) | LLM observability, tracing, prompt management, and evaluation |
| [Milvus](https://milvus.io/) | Vector database for embeddings and similarity search |
| [MCP Gateway Registry](https://github.com/agentic-community/mcp-gateway-registry) | Discovery and management of Model Context Protocol servers |
| [Karpenter](https://karpenter.sh/) | Kubernetes node autoscaling |
| [ArgoCD](https://argo-cd.readthedocs.io/) | GitOps continuous delivery |

## Prerequisites

### Domain and Certificate Setup

GitLab requires a valid TLS certificate, which requires owning a domain. You can use a subdomain from an existing domain.

1. **Create a Route53 Hosted Zone**

   Follow the [AWS documentation to create a hosted zone](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/CreatingHostedZone.html). For a subdomain, name it following the pattern `subdomain.domain.tld`.

2. **(Optional) Configure as Subdomain**

   If using a subdomain, add the hosted zone as a [subdomain](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/CreatingNewSubdomain.html) to your main domain.

3. **Create an ACM Certificate**

   Follow the [ACM documentation to create a certificate](https://docs.aws.amazon.com/acm/latest/userguide/gs-acm-request-public.html) for your domain.

### Tools Required

- AWS CLI configured with appropriate permissions
- Terraform >= 1.0
- kubectl
- Helm >= 3.0

## Deployment

### Step 1: Clone and Navigate

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/agents-on-eks
```

### Step 2: Configure Variables

Edit `terraform/blueprint.tfvars` to set your domain:

```hcl
name                        = "aioeks-agents"
enable_langfuse             = true
enable_gitlab               = true
enable_external_dns         = true
enable_milvus               = true
enable_mcp_gateway_registry = true
max_user_namespaces         = 16384
acm_certificate_domain      = "agents.example.com"  # Update with your domain
allowed_inbound_cidrs       = "0.0.0.0/0"           # Restrict inbound IPs
```

### Step 3: Deploy

```bash
./install.sh
```

Deployment takes approximately 20 minutes.

### Step 4: Configure kubectl

After deployment, configure kubectl to access your cluster:

```bash
aws eks update-kubeconfig --name aioeks-agents --region us-west-2
```

## Accessing Services

### GitLab

GitLab will be available at `https://gitlab.<your-domain>`. Retrieve the root password:

```bash
kubectl get secret gitlab-gitlab-initial-root-password -n gitlab -o jsonpath='{.data.password}' | base64 -d
```

### Langfuse

Access Langfuse through port-forwarding:

```bash
kubectl port-forward svc/langfuse 3000:3000 -n langfuse
```

Then open `http://localhost:3000` in your browser.

### Milvus

Connect to Milvus from within the cluster at `milvus.milvus.svc.cluster.local:19530`.

### MCP Gateway Registry

The MCP Gateway Registry will be available at `https://mcpregistry.<your-domain>`.

## Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `name` | Cluster name | `aioeks-agents` |
| `region` | AWS region | `us-west-2` |
| `eks_cluster_version` | EKS version | `1.34` |
| `acm_certificate_domain` | Domain for TLS certificates | `""` (required) |
| `allowed_inbound_cidrs` | CIDR ranges allowed through load balancer | `0.0.0.0/0` |
| `enable_langfuse` | Deploy Langfuse | `true` |
| `enable_gitlab` | Deploy GitLab | `true` |
| `enable_milvus` | Deploy Milvus | `true` |
| `enable_mcp_gateway_registry` | Deploy MCP Gateway Registry | `true` |
| `enable_external_dns` | Enable External DNS for Route53 | `true` |

### Restricting Inbound Access

The `allowed_inbound_cidrs` variable controls which IP ranges can access services through the load balancer. Restrict this to your organization's IP ranges:

```hcl
allowed_inbound_cidrs = "10.0.0.0/8,192.168.1.0/24"
```

Ensure the CIDR includes your developer IPs and GitLab Runner node IPs for CI/CD pipelines.

## Cleanup

To destroy the infrastructure:

```bash
cd terraform/_LOCAL
./cleanup.sh
```

## Next Steps

- Configure GitLab CI/CD pipelines for your agent code
- Set up Langfuse projects and API keys for tracing
- Create Milvus collections for your embedding storage
- Register MCP servers in the gateway registry
