---
title: NVIDIA Dynamo on Amazon EKS
sidebar_position: 8
---

import CollapsibleContent from '../../../../src/components/CollapsibleContent';

:::warning
Deployment of ML models on EKS requires access to GPUs or Neuron instances. If your deployment isn't working, it's often due to missing access to these resources. Also, some deployment patterns rely on Karpenter autoscaling and static node groups; if nodes aren't initializing, check the logs for Karpenter or Node groups to resolve the issue.
:::

:::info
NVIDIA Dynamo is a cloud-native platform for deploying and managing AI inference graphs at scale. This implementation provides complete infrastructure setup with enterprise-grade monitoring and scalability on Amazon EKS.
:::

# NVIDIA Dynamo on Amazon EKS

:::warning Active Development
This NVIDIA Dynamo blueprint is currently in **active development**. We are continuously improving the user experience and functionality. Features, configurations, and deployment processes may change between releases as we iterate and enhance the implementation based on user feedback and best practices.

Please expect iterative improvements in upcoming releases. If you encounter any issues or have suggestions for improvements, please feel free to open an issue or contribute to the project.
:::

## Quick Start

**Want to get started immediately?** Here's the minimal command sequence:

```bash
# 1. Clone and navigate
git clone https://github.com/awslabs/ai-on-eks.git && cd ai-on-eks/infra/nvidia-dynamo

# 2. Deploy everything (15-30 minutes)
./install.sh

# 3. Build base image and deploy inference
cd ../../blueprints/inference/nvidia-dynamo
source dynamo_env.sh
./build-base-image.sh vllm --push
./deploy.sh

# 4. Test your deployment
./test.sh
```

**Prerequisites**: AWS CLI, kubectl, docker, terraform, earthly, python3.10+, git ([detailed setup below](#prerequisites))

---

## What is NVIDIA Dynamo?

NVIDIA Dynamo is an open-source inference framework designed to optimize performance and scalability for large language models (LLMs) and generative AI applications.

### What is an Inference Graph?

An **inference graph** is a computational workflow that defines how AI models process data through interconnected nodes, enabling complex multi-step AI operations like:
- **LLM chains**: Sequential processing through multiple language models
- **Multimodal processing**: Combining text, image, and audio processing
- **Custom inference pipelines**: Tailored workflows for specific AI applications
- **Disaggregated serving**: Separating prefill and decode phases for optimal resource utilization

## Overview

This blueprint uses the **official NVIDIA Dynamo Helm charts** from the dynamo source repository, with additional shell scripts and Terraform automation to simplify the deployment process on Amazon EKS.

### Deployment Approach

**Why This Setup Process?**
While this implementation involves multiple steps, it provides several advantages over a simple Helm-only deployment:

- **Complete Infrastructure**: Automatically provisions VPC, EKS cluster, ECR repositories, and monitoring stack
- **Production Ready**: Includes enterprise-grade security, monitoring, and scalability features
- **AWS Integration**: Leverages EKS autoscaling, EFA networking, and AWS services
- **Customizable**: Allows fine-tuning of GPU node pools, networking, and resource allocation
- **Reproducible**: Infrastructure as Code ensures consistent deployments across environments

**For Simpler Deployments**: If you already have an EKS cluster and prefer a minimal setup, you can use the Dynamo Helm charts directly from the source repository. This blueprint provides the full production-ready experience.

As LLMs and generative AI applications become increasingly prevalent, the demand for efficient, scalable, and low-latency inference solutions has grown. Traditional inference systems often struggle to meet these demands, especially in distributed, multi-node environments. NVIDIA Dynamo addresses these challenges by offering innovative solutions to optimize performance and scalability with support for AWS services such as Amazon S3, Elastic Fabric Adapter (EFA), and Amazon EKS.

### Key Features

**Performance Optimizations:**
- **Disaggregated Serving**: Separates prefill and decode phases across different GPUs for optimal resource utilization
- **Dynamic GPU Scheduling**: Intelligent resource allocation based on real-time demand through the NVIDIA Dynamo Planner
- **Smart Request Routing**: Minimizes KV cache recomputation by routing requests to workers with relevant cached data
- **Accelerated Data Transfer**: Low-latency communication via NVIDIA NIXL library
- **Efficient KV Cache Management**: Intelligent offloading across memory hierarchies with the KV Cache Block Manager

**Infrastructure Ready:**
- **Inference Engine Agnostic**: Supports TensorRT-LLM, vLLM, SGLang, and other runtimes
- **Modular Design**: Pick and choose components that fit your existing AI stack
- **Enterprise Grade**: Complete monitoring, logging, and security integration
- **Amazon EKS Optimized**: Leverages EKS autoscaling, GPU support, and AWS services

## Architecture

The deployment uses Amazon EKS with the following components:

![NVIDIA Dynamo Architecture](https://github.com/ai-dynamo/dynamo/blob/main/docs/images/architecture.png?raw=true)

**Key Components:**
- **VPC and Networking**: Standard VPC with EFA support for low-latency inter-node communication
- **EKS Cluster**: Managed Kubernetes with GPU-enabled node groups using Karpenter
- **Dynamo Platform**: Operator, API Store, and supporting services (NATS, PostgreSQL, MinIO)
- **Monitoring Stack**: Prometheus, Grafana, and AI/ML observability
- **Storage**: Amazon EFS for shared model storage and caching

## Prerequisites

**System Requirements**: Ubuntu 22.04 or 24.04 (NVIDIA Dynamo officially supports only these versions)

Install the following tools on your setup host (recommended: EC2 instance t3.xlarge or higher with EKS and ECR permissions):

- **AWS CLI**: Configured with appropriate permissions ([installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
- **kubectl**: Kubernetes command-line tool ([installation guide](https://kubernetes.io/docs/tasks/tools/install-kubectl/))
- **helm**: Kubernetes package manager ([installation guide](https://helm.sh/docs/intro/install/))
- **terraform**: Infrastructure as code tool ([installation guide](https://learn.hashicorp.com/tutorials/terraform/install-cli))
- **docker**: With buildx and user needs docker permissions ([installation guide](https://docs.docker.com/get-docker/))
- **earthly**: Multi-platform build automation tool used by NVIDIA Dynamo for reproducible container builds ([installation guide](https://earthly.dev/get-earthly))
- **Python 3.10+**: With pip and venv ([installation guide](https://www.python.org/downloads/))
- **git**: Version control ([installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))
- **EKS Cluster**: Version 1.33 (tested and supported)

<CollapsibleContent header={<h2><span>Deploying the Solution</span></h2>}>

Complete the following steps to deploy NVIDIA Dynamo on Amazon EKS:

### Step 1: Clone the Repository

```bash
git clone https://github.com/awslabs/ai-on-eks.git && cd ai-on-eks
```

### Step 2: Deploy Infrastructure and Platform

Navigate to the infrastructure directory and run the installation script:

```bash
cd infra/nvidia-dynamo
./install.sh
```

This command provisions your complete environment:
- **VPC**: Subnets, security groups, NAT gateways, and internet gateway
- **EKS Cluster**: With GPU-enabled node groups using Karpenter
- **ECR Repositories**: For Dynamo container images
- **Monitoring Stack**: Prometheus, Grafana, and AI/ML observability
- **Dynamo Platform**: Deploys using official NVIDIA Dynamo Helm charts (Operator, API Store, NATS, PostgreSQL, MinIO)

**Duration**: 15-30 minutes

### Step 3: Build Base Images

**Why Custom Image Builds?**
Currently, official NVIDIA Dynamo container images are not yet available from NVIDIA's public registries. This blueprint builds the required images using the official Dynamo build process and pushes them to your private ECR repositories. Future releases will include pre-built images from NVIDIA.

Build and push the base images for your chosen inference framework:

```bash
cd blueprints/inference/nvidia-dynamo
source dynamo_env.sh   # Generated by install.sh with AWS credentials

# Build vLLM base image (recommended for most LLMs)
./build-base-image.sh vllm --push

# Optional: Build other framework images
./build-base-image.sh tensorrtllm --push
./build-base-image.sh sglang --push
```

**Framework Options:**
- **vLLM**: Best for most LLMs, supports many model formats
- **TensorRT-LLM**: Optimized for NVIDIA GPUs, fastest inference
- **SGLang**: Structured generation for complex prompting

**Build Process:**
- Uses the official Dynamo `container/build.sh` script
- Leverages Earthly for reproducible, multi-platform builds
- Configures CUDA drivers and framework-specific dependencies
- Pushes to your private ECR repositories for secure access

### Step 4: Deploy Inference Graphs

Deploy your inference service using the interactive deployment script:

```bash
./deploy.sh
```

The interactive menu will guide you through:
1. **Example Type**: Choose between hello-world or llm
2. **LLM Architecture**: Select from agg, disagg, agg_router, disagg_router, multinode options
3. **Automatic Configuration**: Sets up monitoring and service exposure

**Architecture Options:**
- **agg**: Aggregated - single node processing
- **disagg**: Disaggregated - separate prefill/decode phases
- **agg_router**: Aggregated with smart routing
- **disagg_router**: Disaggregated with smart routing (recommended)
- **multinode**: Multi-node setups for large models

The deployment creates monitoring resources (Service and ServiceMonitor) automatically.

</CollapsibleContent>

## Test and Validate

### Automated Testing

Use the built-in test script to validate your deployment:

```bash
./test.sh
```

This script:
- Starts port forwarding to the frontend service
- Tests health check, metrics, and `/v1/models` endpoints
- Runs sample inference requests to verify functionality

### Manual Testing

Access your deployment directly:

```bash
kubectl port-forward svc/<frontend-service> 3000:3000 -n dynamo-cloud &

curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
        {"role": "user", "content": "Explain what a Q-Bit is in quantum computing."}
    ],
    "max_tokens": 2000,
    "temperature": 0.7,
    "stream": false
}'
```

**Expected Output:**
```json
{
  "id": "1918b11a-6d98-4891-bc84-08f99de70fd0",
  "choices": [
    {
      "index": 0,
      "message": {
        "content": "A Q-bit, or qubit, is the basic unit of quantum information...",
        "role": "assistant"
      },
      "finish_reason": "stop"
    }
  ],
  "created": 1752018267,
  "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
  "object": "chat.completion"
}
```

## Monitor and Observe

### Grafana Dashboard

Access Grafana for visualization (default port 3000):

```bash
kubectl port-forward -n kube-prometheus-stack svc/kube-prometheus-stack-grafana 3000:80
```

### Prometheus Metrics

Access Prometheus for metrics collection (port 9090):

```bash
kubectl port-forward -n kube-prometheus-stack svc/prometheus 9090:80
```

### Automatic Monitoring

The deployment automatically creates:
- **Service**: Exposes inference graphs for API calls and metrics
- **ServiceMonitor**: Configures Prometheus to scrape metrics
- **Dashboards**: Pre-configured Grafana dashboards for inference monitoring

## Advanced Configuration

### ECR Authentication

The deployment uses **IRSA (IAM Roles for Service Accounts)** for secure ECR access:

- **Primary Method**: IRSA eliminates credential rotation
- **Fallback Method**: ECR token refresh CronJob (legacy mode)
- **Security**: Service accounts automatically authenticate to ECR
- **No Secrets**: No long-lived credentials stored in Kubernetes

**Note**: AWS Pod Identity is available as an alternative to IRSA (GA since April 2024), but IRSA remains the recommended approach due to its maturity, wide adoption, and proven reliability in production environments.

### Custom Model Deployment

To deploy custom models, modify the configuration files in `dynamo/examples/llm/configs/`:

1. **Choose Architecture**: Select based on model size and requirements
2. **Update Configuration**: Edit the appropriate YAML file
3. **Set Model Parameters**: Update `model` and `served_model_name` fields
4. **Configure Resources**: Adjust GPU allocation and memory settings

**Example for DeepSeek-R1 70B model:**

```yaml
Common:
  model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
  max-model-len: 32768
  tensor-parallel-size: 4

Frontend:
  served_model_name: deepseek-ai/DeepSeek-R1-Distill-Llama-70B

VllmWorker:
  ServiceArgs:
    resources:
      gpu: '4'
```

### Karpenter Node Pools

The deployment can optionally use custom Karpenter node pools optimized for NVIDIA Dynamo:

- **C7i CPU Pools**: For general compute and BuildKit (newer than base M5 instances)
- **G6 GPU Pools**: For inference workloads with NVIDIA L4 GPUs
- **Higher Priority**: Weight 100 vs base addons weight 50 for priority scheduling
- **BuildKit Support**: User namespace configuration for container builds
- **EFA Support**: Low-latency networking for multi-node setups

**Note**: Custom node pools are disabled by default. The base infrastructure provides existing Karpenter node pools (G6 GPU, G5 GPU, M5 CPU) that work well for most Dynamo workloads. Enable custom pools only if you need BuildKit support or higher scheduling priority.

### Configuration Options

Modify `terraform/blueprint.tfvars` for customization:

```hcl
# Enable custom node pools (optional - disabled by default)
enable_custom_karpenter_nodepools = true

# Choose AMI (AL2023 recommended)
use_bottlerocket = false

# Resource limits
karpenter_cpu_limits = 10000
karpenter_memory_limits = 10000
```

## Troubleshooting

### Common Issues

1. **GPU Nodes Not Available**: Check Karpenter logs and instance availability
2. **Image Pull Errors**: Verify ECR repositories and image push success
3. **Pod Failures**: Check resource limits and cluster capacity
4. **Deployment Timeouts**: Ensure base images are built and available

### Debug Commands

```bash
# Check cluster status
kubectl get nodes
kubectl get pods -n dynamo-cloud

# View logs
kubectl logs -n dynamo-cloud deployment/dynamo-operator
kubectl logs -n dynamo-cloud deployment/dynamo-api-store

# Check deployments
source dynamo_venv/bin/activate
dynamo deployment list --endpoint "$DYNAMO_CLOUD"
```

### Performance Optimization

- **Model Size Guidelines**: Use appropriate architecture for model parameters
- **Resource Allocation**: Match GPU count to model requirements
- **Network Configuration**: Ensure EFA is enabled for multi-node setups
- **Storage**: Use EFS for shared model storage and caching

## Alternative Deployment Options

### For Existing EKS Clusters

If you already have an EKS cluster with GPU nodes and prefer a simpler approach:

1. **Direct Helm Installation**: Use the official NVIDIA Dynamo Helm charts directly from the [dynamo source repository](https://github.com/ai-dynamo/dynamo)
2. **Manual Setup**: Follow the upstream NVIDIA Dynamo documentation for Kubernetes deployment
3. **Custom Integration**: Integrate Dynamo components into your existing infrastructure

### Why Use This Blueprint?

This blueprint is designed for users who want:
- **Complete Infrastructure**: End-to-end setup from VPC to running inference
- **Production Readiness**: Enterprise-grade monitoring, security, and scalability
- **AWS Integration**: Optimized for EKS, ECR, EFA, and other AWS services
- **Best Practices**: Follows ai-on-eks patterns and AWS recommendations

## Repository Information

- **Repository**: [awslabs/ai-on-eks](https://github.com/awslabs/ai-on-eks)
- **Documentation**: [Complete NVIDIA Dynamo Blueprint](https://github.com/awslabs/ai-on-eks/tree/main/blueprints/inference/nvidia-dynamo)
- **NVIDIA Dynamo**: [Official Documentation](https://docs.nvidia.com/dynamo/)
- **Dynamo Source**: [NVIDIA Dynamo Repository](https://github.com/ai-dynamo/dynamo)

## Next Steps

1. **Explore Examples**: Check the examples folder in the GitHub repository
2. **Scale Deployments**: Configure multi-node setups for larger models
3. **Integrate Applications**: Connect your applications to the inference endpoints
4. **Monitor Performance**: Use Grafana dashboards for ongoing monitoring
5. **Optimize Costs**: Implement auto-scaling and resource optimization

## Clean Up

When you're finished with your NVIDIA Dynamo deployment, remove all resources using the cleanup script:

```bash
cd infra/nvidia-dynamo
./cleanup.sh
```

This safely destroys the NVIDIA Dynamo deployments and infrastructure components in the correct order, including:
- Dynamo platform components and workloads
- Kubernetes resources and namespaces
- ECR repositories and container images
- Terraform-managed infrastructure (EKS cluster, VPC, etc.)

The cleanup script ensures proper resource cleanup to avoid any lingering costs.

This deployment provides a production-ready NVIDIA Dynamo environment on Amazon EKS with enterprise-grade features including Karpenter automatic scaling, EFA networking, and seamless AWS service integration.
