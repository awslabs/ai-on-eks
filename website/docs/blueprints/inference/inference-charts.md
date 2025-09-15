---
sidebar_label: Inference Charts
---

# AI on EKS Inference Charts

The AI on EKS Inference Charts provide a streamlined Helm-based approach to deploy AI/ML inference workloads on both GPU
and AWS Neuron (Inferentia/Trainium) hardware. This chart supports multiple deployment configurations and comes with
pre-configured values for popular models.

## Overview

The inference charts support the following deployment types:

- **GPU-based VLLM deployments** - Single-node VLLM inference
- **GPU-based Ray-VLLM deployments** - Distributed VLLM inference with Ray
- **Neuron-based VLLM deployments** - VLLM inference on AWS Inferentia chips
- **Neuron-based Ray-VLLM deployments** - Distributed VLLM inference with Ray on Inferentia

## Prerequisites

Before deploying the inference charts, ensure you have:

- Amazon EKS cluster with GPU or AWS Neuron nodes ([JARK-stack](../../infra/jark.md) for a quick start)
- Helm 3.0+
- For GPU deployments: NVIDIA device plugin installed
- For Neuron deployments: AWS Neuron device plugin installed
- Hugging Face Hub token (stored as a Kubernetes secret)

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
# Deploy Llama 3.2 1B on GPU with VLLM
helm install llama-inference ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-32-1b-vllm.yaml

# Deploy DeepSeek R1 Distill on GPU with Ray-VLLM
helm install deepseek-inference ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml
```

## Supported Models

The inference charts include pre-configured values files for the following models:

### GPU Models

| Model                         | Size | Framework | Values File                                             |
|-------------------------------|------|-----------|---------------------------------------------------------|
| **DeepSeek R1 Distill Llama** | 8B   | Ray-VLLM  | `values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml` |
| **Llama 3.2**                 | 1B   | VLLM      | `values-llama-32-1b-vllm.yaml`                          |
| **Llama 3.2**                 | 1B   | Ray-VLLM  | `values-llama-32-1b-ray-vllm.yaml`                      |
| **Llama 4 Scout**             | 17B  | VLLM      | `values-llama-4-scout-17b-vllm.yaml`                    |
| **Mistral Small**             | 24B  | Ray-VLLM  | `values-mistral-small-24b-ray-vllm.yaml`                |

### Neuron Models (AWS Inferentia/Trainium)

| Model                         | Size | Framework | Values File                                            |
|-------------------------------|------|-----------|--------------------------------------------------------|
| **DeepSeek R1 Distill Llama** | 8B   | VLLM      | `values-deepseek-r1-distill-llama-8b-vllm-neuron.yaml` |
| **Llama 2**                   | 13B  | Ray-VLLM  | `values-llama-2-13b-ray-vllm-neuron.yaml`              |
| **Llama 3**                   | 70B  | Ray-VLLM  | `values-llama-3-70b-ray-vllm-neuron.yaml`              |
| **Llama 3.1**                 | 8B   | VLLM      | `values-llama-31-8b-vllm-neuron.yaml`                  |
| **Llama 3.1**                 | 8B   | Ray-VLLM  | `values-llama-31-8b-ray-vllm-neuron.yaml`              |

## Deployment Examples

### GPU Deployments

#### Deploy Llama 3.2 1B with VLLM

```bash
helm install llama32-vllm ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-32-1b-vllm.yaml
```

#### Deploy DeepSeek R1 Distill with Ray-VLLM

```bash
helm install deepseek-ray ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml
```

#### Deploy Mistral Small 24B with Ray-VLLM

```bash
helm install mistral-ray ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-mistral-small-24b-ray-vllm.yaml
```

### Neuron Deployments

#### Deploy Llama 3.1 8B with VLLM on Inferentia

```bash
helm install llama31-neuron ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-31-8b-vllm-neuron.yaml
```

#### Deploy Llama 3 70B with Ray-VLLM on Inferentia

```bash
helm install llama3-70b-neuron ./blueprints/inference/inference-charts \
  --values ./blueprints/inference/inference-charts/values-llama-3-70b-ray-vllm-neuron.yaml
```

## Configuration Options

### Key Parameters

The chart provides extensive configuration options. Here are the most important parameters:

| Parameter                                   | Description                           | Default                     |
|---------------------------------------------|---------------------------------------|-----------------------------|
| `inference.accelerator`                     | Accelerator type (`gpu` or `neuron`)  | `gpu`                       |
| `inference.framework`                       | Framework type (`vllm` or `ray-vllm`) | `vllm`                      |
| `inference.serviceName`                     | Name of the inference service         | `inference`                 |
| `inference.modelServer.deployment.replicas` | Number of replicas                    | `1`                         |
| `modelParameters.modelId`                   | Model ID from Hugging Face Hub        | `NousResearch/Llama-3.2-1B` |
| `modelParameters.gpuMemoryUtilization`      | GPU memory utilization                | `0.8`                       |
| `modelParameters.maxModelLen`               | Maximum model sequence length         | `8192`                      |
| `modelParameters.tensorParallelSize`        | Tensor parallel size                  | `1`                         |
| `service.type`                              | Service type                          | `ClusterIP`                 |
| `service.port`                              | Service port                          | `8000`                      |

### Custom Deployment

Create your own values file for custom configurations:

```yaml
inference:
  accelerator: gpu  # or neuron
  framework: vllm   # or ray-vllm
  serviceName: custom-inference
  modelServer:
    deployment:
      replicas: 2
      resources:
        gpu:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1

modelParameters:
  modelId: "your-custom-model-id"
  gpuMemoryUtilization: "0.9"
  maxModelLen: "4096"
  tensorParallelSize: "1"
```

Deploy with custom values:

```bash
helm install custom-inference ./blueprints/inference/inference-charts \
  --values custom-values.yaml
```

## API Endpoints

Once deployed, the service exposes OpenAI-compatible API endpoints:

- **`/v1/models`** - List available models
- **`/v1/completions`** - Text completion API
- **`/v1/chat/completions`** - Chat completion API
- **`/metrics`** - Prometheus metrics endpoint

### Example API Usage

Note: These deployments do not create an ingress, you will need to `kubectl port-forward` to test from your machine,
eg (for deepseek):

```bash
kubectl get svc | grep deepseek
# Note the service name for deepseek, in this case deepseekr1-dis-lllama-8b-ray-vllm-gpu-ray-vllm
kubectl port-forward svc/deepseekr1-dis-llama-8b-ray-vllm-gpu-ray-vllm 8000
```

```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model-name",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Monitoring and Observability

The charts include built-in observability features:

- **Fluent Bit** for log collection
- **Prometheus metrics** for monitoring
- **Grafana dashboards** for visualizations

Access metrics at the `/metrics` endpoint of your deployed service.

## Troubleshooting

### Common Issues

1. **Pod stuck in Pending state**
    - Check if GPU/Neuron nodes are available
    - Verify resource requests match available hardware

2. **Model download failures**
    - Ensure Hugging Face token is correctly configured
    - Check network connectivity to Hugging Face Hub

3. **Out of memory errors**
    - Adjust `gpuMemoryUtilization` parameter
    - Consider using tensor parallelism for larger models

### Logs

Check deployment logs:

```bash
kubectl logs -l app=inference-server
```

For Ray deployments, check Ray cluster status:

```bash
kubectl exec -it <ray-head-pod> -- ray status
```

## Next Steps

- Explore [GPU-specific configurations](/docs/category/gpu-inference-on-eks) for GPU deployments
- Learn about [Neuron-specific configurations](/docs/category/neuron-inference-on-eks) for Inferentia deployments
