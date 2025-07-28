---
sidebar_label: Overview
sidebar_position: 1
---

# Inference on EKS

AI on EKS provides comprehensive solutions for deploying AI/ML inference workloads on Amazon EKS, supporting both GPU and AWS Neuron (Inferentia/Trainium) hardware configurations.

## Quick Start Options

### 🚀 Inference Charts (Recommended)
Get started quickly with our pre-configured Helm charts that support multiple models and deployment patterns:

- **[Inference Charts](./inference-charts.md)** - Streamlined Helm-based deployments with pre-configured values for popular models
- Supports both GPU and Neuron hardware
- Includes VLLM and Ray-VLLM frameworks
- Pre-configured for 10+ popular models including Llama, DeepSeek, and Mistral

## Hardware-Specific Guides

### GPU Deployments
Explore GPU-specific inference solutions:

- [DeepSeek-R1 with Ray and vLLM](./GPUs/ray-vllm-deepseek.md)
- [NVIDIA NIM with Llama3](./GPUs/nvidia-nim-llama3.md)
- [NVIDIA NIM Operator](./GPUs/nvidia-nim-operator.md)
- [vLLM with NVIDIA Triton Server](./GPUs/vLLM-NVIDIATritonServer.md)
- [vLLM with Ray Serve](./GPUs/vLLM-rayserve.md)
- [Stable Diffusion on GPUs](./GPUs/stablediffusion-gpus.md)
- [AIBrix with DeepSeek](./GPUs/aibrix-deepseek-distill.md)

### Neuron Deployments (AWS Inferentia)
Leverage AWS Inferentia chips for cost-effective inference:

- [Llama2 on Inferentia2](./Neuron/llama2-inf2.md)
- [Llama3 on Inferentia2](./Neuron/llama3-inf2.md)
- [Mistral 7B on Inferentia2](./Neuron/Mistral-7b-inf2.md)
- [Ray Serve High Availability](./Neuron/rayserve-ha.md)
- [vLLM with Ray on Inferentia2](./Neuron/vllm-ray-inf2.md)
- [Stable Diffusion on Inferentia2](./Neuron/stablediffusion-inf2.md)

## Architecture Overview

AI on EKS inference solutions support multiple deployment patterns:

- **Single-node inference** with vLLM
- **Distributed inference** with Ray-vLLM
- **Production-ready** deployments with load balancing
- **Auto-scaling** capabilities
- **Observability** and monitoring integration

## Choosing the Right Approach

| Use Case | Recommended Solution                                       | Benefits |
|----------|------------------------------------------------------------|----------|
| **Quick prototyping** | [Inference Charts](./inference-charts.md)                  | Pre-configured, fast deployment |
| **GPU** | [GPU-specific guides](/docs/category/gpu-inference-on-eks) | GPU-based inference |
| **Neuron** | [Neuron guides](/docs/category/neuron-inference-on-eks)   | Inferentia-based inference |

## Next Steps

1. **Start with [Inference Charts](./inference-charts.md)** for the fastest path to deployment
2. **Explore hardware-specific guides** for optimized configurations
3. **Set up monitoring and observability** for production workloads
