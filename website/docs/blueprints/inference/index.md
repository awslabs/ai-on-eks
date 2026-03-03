---
sidebar_position: 1
---

# Inference on EKS

Deploy and run Large Language Models (LLMs) and other AI models on Amazon EKS.

## What's in This Section

This section provides practical deployment guides and Helm charts for running inference workloads on EKS. Whether you're deploying open-source LLMs, diffusion models, or custom AI models, you'll find ready-to-use configurations and step-by-step instructions.

---

## [Inference Charts](./inference-charts.md)

Helm charts for deploying popular AI models on EKS with pre-configured values for optimal performance.

**What You Get:**
- Ready-to-deploy Helm charts for vLLM, Ray-vLLM, Triton, and Diffusers
- Pre-configured values files for popular models (Llama, DeepSeek, Mistral, Stable Diffusion, and more)
- Support for both GPU (NVIDIA) and Neuron (AWS Inferentia/Trainium) deployments
- Configurations with health checks, autoscaling, and monitoring

**Use Cases:**
- Quick deployment of open-source LLMs
- Standardized deployment patterns across your organization
- Reference implementations for custom model deployments

[Explore Inference Charts →](./inference-charts.md)

---

## Framework-Specific Deployment Guides

Detailed guides for deploying models with deep dive into specific frameworks on EKS, organized by hardware type.

### GPU Deployments

Step-by-step guides for deploying models on NVIDIA GPUs:

- **[AIBrix DeepSeek Distill](/docs/blueprints/inference/framework-guides/GPUs/aibrix-deepseek-distill)** - Deploy DeepSeek R1 Distill Llama 8B with AIBrix optimization
- **[NVIDIA Dynamo](/docs/blueprints/inference/framework-guides/GPUs/nvidia-dynamo)** - Deploy models with NVIDIA's Dynamo framework
- **[NVIDIA NIM Llama 3](/docs/blueprints/inference/framework-guides/GPUs/nvidia-nim-llama3)** - Deploy Llama 3 using NVIDIA NIM
- **[NVIDIA NIM Operator](/docs/blueprints/inference/framework-guides/GPUs/nvidia-nim-operator)** - Kubernetes operator for NVIDIA NIM deployments
- **[vLLM with NVIDIA Triton Server](/docs/blueprints/inference/framework-guides/GPUs/vLLM-NVIDIATritonServer)** - Inference with Triton and vLLM
- **[vLLM with Ray Serve](/docs/blueprints/inference/framework-guides/GPUs/vLLM-rayserve)** - Scalable inference with Ray Serve and vLLM

### Neuron Deployments

Step-by-step guides for deploying models on AWS Inferentia and Trainium:

- **[Mistral 7B on Inf2](/docs/blueprints/inference/framework-guides/Neuron/Mistral-7b-inf2)** - Deploy Mistral 7B on AWS Inferentia 2
- **[Llama 2 on Inf2](/docs/blueprints/inference/framework-guides/Neuron/llama2-inf2)** - Deploy Llama 2 13B on AWS Inferentia 2
- **[Llama 3 on Inf2](/docs/blueprints/inference/framework-guides/Neuron/llama3-inf2)** - Deploy Llama 3 on AWS Inferentia 2
- **[Ray Serve High Availability](/docs/blueprints/inference/framework-guides/Neuron/rayserve-ha)** - Deploy highly available Ray Serve on Neuron
- **[Stable Diffusion on Inf2](/docs/blueprints/inference/framework-guides/Neuron/stablediffusion-inf2)** - Deploy Stable Diffusion on AWS Inferentia 2
- **[vLLM Ray on Inf2](/docs/blueprints/inference/framework-guides/Neuron/vllm-ray-inf2)** - Deploy vLLM with Ray on AWS Inferentia 2



---

## Getting Started

1. **Set up your infrastructure** - Start with the [Inference-Ready Cluster](/docs/infra/inference/inference-ready-cluster) to provision an EKS cluster optimized for AI/ML workloads

2. **Choose your deployment method**:
   - For quick deployments with popular models → Use [Inference Charts](./inference-charts.md)
   - For specific frameworks or custom configurations → See Framework-Specific Guides above

3. **Optimize your deployment** - Apply best practices from the [Guidance section](/docs/guidance/) to improve performance and reduce costs

---

## Need Help?

- **Infrastructure Setup**: See [Inference Infrastructure](/docs/infra/inference/) for cluster setup and configuration
- **Optimization**: Check the [Guidance section](/docs/guidance/) for performance tuning and best practices
- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/awslabs/ai-on-eks/issues)
- **Community**: Join discussions on [GitHub Discussions](https://github.com/awslabs/ai-on-eks/discussions)
