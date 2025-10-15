---
sidebar_label: Inference Charts
---

# AI on EKS Inference Charts

The AI on EKS Inference Charts provide a streamlined Helm-based approach to deploy AI/ML inference workloads on both GPU and AWS Neuron (Inferentia/Trainium) hardware. This chart supports multiple deployment configurations and comes with pre-configured values for popular models.

:::info Advanced Usage
For detailed configuration options, advanced deployment scenarios, and comprehensive parameter documentation, see the [complete README](https://github.com/awslabs/ai-on-eks/blob/main/blueprints/inference/inference-charts/README.md).
:::

## Overview

The inference charts support multiple deployment frameworks:

- **VLLM** - Single-node inference with fast startup
- **Ray-VLLM** - Distributed inference with autoscaling capabilities
- **Triton-VLLM** - Production-ready inference server with advanced features
- **AIBrix** - VLLM with AIBrix-specific configurations
- **LeaderWorkerSet-VLLM** - Multi-node inference for large models
- **Diffusers** - Hugging Face Diffusers for image generation

Both GPU and AWS Neuron (Inferentia/Trainium) accelerators are supported across these frameworks.

## Prerequisites

Before deploying the inference charts, ensure you have:

- Amazon EKS cluster with GPU or AWS Neuron
  nodes ([inference-ready cluster](../../infra/inference-ready-cluster.md) for a quick start)
- Helm 3.0+
- For GPU deployments: NVIDIA device plugin installed
- For Neuron deployments: AWS Neuron device plugin installed
- For LeaderWorkerSet deployments: LeaderWorkerSet CRD installed
- Hugging Face Hub token (stored as a Kubernetes secret named `hf-token`)
- For Ray: KubeRay Infrastructure
- For AIBrix: AIBrix Infrastructure

## Quick Start

### 1. Create Hugging Face Token Secret

Create a Kubernetes secret with your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens):

```bash
kubectl create secret generic hf-token --from-literal=token=your_huggingface_token
```

### 2. Deploy a Pre-configured Model

Choose from the available pre-configured models and deploy:

:::warning

These deployments will need GPU/Neuron resources which need to
be [enabled](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) and cost more than CPU only
instances.

:::

```bash
# Deploy Llama 3.2 1B on GPU with vLLM
helm install llama-inference ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-32-1b-vllm.yaml

# Deploy DeepSeek R1 Distill on GPU with Ray-vLLM
helm install deepseek-inference ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml
```

## Supported Models

The inference charts include pre-configured values files for popular models across different categories:

### Language Models
- **DeepSeek R1 Distill Llama 8B** - Advanced reasoning model
- **Llama 3.2 1B** - Lightweight language model
- **Llama 4 Scout 17B** - Mid-size language model
- **Mistral Small 24B** - Efficient large language model
- **GPT OSS 20B** - Open-source GPT variant

### Diffusion Models
- **FLUX.1 Schnell** - Fast text-to-image generation
- **Stable Diffusion XL** - High-quality image generation
- **Stable Diffusion 3.5** - Latest SD model with enhanced capabilities
- **Kolors** - Artistic image generation
- **OmniGen** - Multi-modal generation

### Neuron-Optimized Models
- **Llama 2 13B** - Optimized for AWS Inferentia
- **Llama 3 70B** - Large model on Inferentia
- **Llama 3.1 8B** - Efficient Inferentia deployment

Each model comes with optimized configurations for different frameworks (VLLM, Ray-VLLM, Triton-VLLM, etc.).

## Deployment Examples

### Language Model Deployments

```bash
# Deploy Llama 3.2 1B with VLLM
helm install llama32-vllm ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-32-1b-vllm.yaml

# Deploy DeepSeek R1 Distill with Ray-VLLM
helm install deepseek-ray ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml

# Deploy Llama 4 Scout 17B with LeaderWorkerSet-VLLM
helm install llama4-lws ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-4-scout-17b-lws-vllm.yaml
```

### Diffusion Model Deployments

```bash
# Deploy FLUX.1 Schnell for image generation
helm install flux-diffusers ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-flux-1-diffusers.yaml

# Deploy Stable Diffusion XL
helm install sdxl-diffusers ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-stable-diffusion-xl-base-1-diffusers.yaml
```

### Neuron Deployments

```bash
# Deploy Llama 3.1 8B on Inferentia
helm install llama31-neuron ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-31-8b-vllm-neuron.yaml

# Deploy Llama 3 70B with Ray-VLLM on Inferentia
helm install llama3-70b-neuron ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-3-70b-ray-vllm-neuron.yaml
```

## Configuration

### Key Parameters

| Parameter                                   | Description                                                   | Default                     |
|---------------------------------------------|---------------------------------------------------------------|-----------------------------|
| `inference.accelerator`                     | Accelerator type (`gpu` or `neuron`)                          | `gpu`                       |
| `inference.framework`                       | Framework (`vllm`, `ray-vllm`, `triton-vllm`, `aibrix`, etc.) | `vllm`                      |
| `inference.serviceName`                     | Name of the inference service                                 | `inference`                 |
| `inference.modelServer.deployment.replicas` | Number of replicas                                            | `1`                         |
| `model`                                     | Model ID from Hugging Face Hub                                | `NousResearch/Llama-3.2-1B` |
| `modelParameters.gpuMemoryUtilization`      | GPU memory utilization                                        | `0.8`                       |
| `modelParameters.maxModelLen`               | Maximum model sequence length                                 | `8192`                      |
| `modelParameters.tensorParallelSize`        | Tensor parallel size                                          | `1`                         |
| `modelParameters.pipelineParallelSize`      | Pipeline parallel size                                        | `1`                         |

### Custom Configuration

Create a custom values file:

```yaml
inference:
  accelerator: gpu  # or neuron
  framework: vllm   # vllm, ray-vllm, triton-vllm, aibrix, lws-vllm, diffusers
  serviceName: my-inference
  modelServer:
    deployment:
      replicas: 1
      instanceType: g5.2xlarge

model: "NousResearch/Llama-3.2-1B"
modelParameters:
  gpuMemoryUtilization: 0.8
  maxModelLen: 8192
  tensorParallelSize: 1
```

Deploy with custom values:

```bash
helm install my-inference ./blueprints/inference/inference-charts \
  --values custom-values.yaml
```

## API Usage

The deployed services expose different API endpoints based on the framework:

### VLLM/Ray-VLLM
- `/v1/models` - List available models
- `/v1/chat/completions` - Chat completion API
- `/v1/completions` - Text completion API
- `/metrics` - Prometheus metrics

### Triton-VLLM
- `/v2/models` - List available models
- `/v2/models/vllm_model/generate` - Model inference
- `/v2/health/ready` - Health checks

### Diffusers
- `/v1/generations` - Image generation API

### Example Usage

Access your service via port-forward:

```bash
kubectl port-forward svc/<service-name> 8000
```

Test the API:

```bash
# Chat completion (VLLM/Ray-VLLM)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Image generation (Diffusers)
curl -X POST http://localhost:8000/v1/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A beautiful sunset over mountains"}'
```

## Troubleshooting

### Common Issues

1. **Pod stuck in Pending state**
   - Check if GPU/Neuron nodes are available
   - Verify resource requests match available hardware
   - For LeaderWorkerSet deployments: Ensure LeaderWorkerSet CRD is installed

2. **Model download failures**
   - Ensure Hugging Face token is correctly configured as secret `hf-token`
   - Check network connectivity to Hugging Face Hub
   - Verify model ID is correct and accessible

3. **Out of memory errors**
   - Adjust `gpuMemoryUtilization` parameter (try reducing from 0.8 to 0.7)
   - Consider using tensor parallelism for larger models
   - For large models, use LeaderWorkerSet or Ray deployments with multiple GPUs

4. **Ray deployment issues**
   - Ensure KubeRay infrastructure is installed
   - Check Ray cluster status and worker connectivity
   - Verify Ray version compatibility

5. **Triton deployment issues**
   - Check Triton server logs for model loading errors
   - Verify model repository configuration
   - Ensure proper health check endpoints are accessible

### Logs

Check deployment logs based on framework:
### Check Logs

```bash
# VLLM deployments
kubectl logs -l app.kubernetes.io/component=<service-name>

# Ray deployments
kubectl logs -l ray.io/node-type=head
kubectl logs -l ray.io/node-type=worker

# LeaderWorkerSet deployments
kubectl logs -l leaderworkerset.sigs.k8s.io/role=leader
```

## Next Steps

- Explore [GPU-specific configurations](/docs/category/gpu-inference-on-eks) for GPU deployments
- Learn about [Neuron-specific configurations](/docs/category/neuron-inference-on-eks) for Inferentia deployments
