# AI on EKS Inference Charts

Chart Name: `ai-on-eks-inference-charts`

This Helm chart provides deployment configurations for AI/ML inference workloads on both GPU and AWS Neuron (
Inferentia/Trainium) hardware.

## Overview

The chart supports the following deployment types:

- GPU-based VLLM deployments
- GPU-based Ray-VLLM deployments
- GPU-based Triton-VLLM deployments
- GPU-based AIBrix deployments
- GPU-based LeaderWorkerSet-VLLM deployments
- GPU-based Diffusers deployments
- Neuron-based VLLM deployments
- Neuron-based Ray-VLLM deployments
- Neuron-based Triton-VLLM deployments (Coming Soon)
- Ray-VLLM deployments with (optional) GCS High Availability

### VLLM vs Ray-VLLM vs LeaderWorkerSet-VLLM

**VLLM Deployments** (`framework: vllm`):

- Direct VLLM deployment using Kubernetes Deployment
- Simpler architecture, faster startup
- Uses `vllm/vllm-openai` image
- Suitable for single-node inference

**Ray-VLLM Deployments** (`framework: ray-vllm`):

- VLLM deployed on Ray Serve for distributed inference
- More complex architecture with head and worker nodes
- Uses `rayproject/ray` image
- Supports autoscaling and distributed workloads
- Includes observability integration with Prometheus and Grafana
- Requires additional parameters: `rayVersion`, `vllmVersion`, `pythonVersion`

**AIBrix Deployments** (`framework: aibrix`):

- VLLM deployment with AIBrix-specific configurations
- Uses `vllm/vllm-openai` image
- Includes additional model labels for AIBrix integration
- Suitable for AIBrix-managed inference workloads

**Triton-VLLM Deployments** (`framework: triton-vllm`):

- VLLM deployed as a backend for NVIDIA Triton Inference Server
- Production-ready inference server with advanced features
- Uses `nvcr.io/nvidia/tritonserver` image for GPU or `public.ecr.aws/neuron/tritonserver` for Neuron
- Supports both HTTP and gRPC protocols
- Includes health checks, metrics, and model repository management
- Compatible with both GPU and AWS Neuron accelerators (Soon)

**LeaderWorkerSet-VLLM Deployments** (`framework: lws-vllm`):

- VLLM deployed using Kubernetes LeaderWorkerSet for multi-node inference
- Simplified distributed architecture with leader and worker pods
- Uses `vllm/vllm-openai` image
- Ideal for large models requiring pipeline parallelism across multiple nodes
- Automatic leader-worker coordination and service discovery
- Requires LeaderWorkerSet CRD to be installed in the cluster

**Diffusers Deployments** (`framework: diffusers`):

- Hugging Face Diffusers library for image generation and diffusion models
- Supports various diffusion pipelines including Stable Diffusion, FLUX, Kolors, and more
- Uses custom diffusers container image optimized for GPU inference
- Ideal for text-to-image, image-to-image, and other generative AI workloads
- Supports multiple pipeline types: `stable-diffusion`, `diffusion`, `kolors`, `stablediffusion3`, `omnigen`

## Prerequisites

- Kubernetes cluster with GPU or AWS Neuron nodes
- Helm 3.0+
- For GPU deployments: NVIDIA device plugin installed
- For Neuron deployments: AWS Neuron device plugin installed
- For LeaderWorkerSet deployments: LeaderWorkerSet CRD installed
- Hugging Face Hub token (stored as a Kubernetes secret named `hf-token`)
- For Ray: KubeRay Infrastructure
- For AIBrix: AIBrix Infrastructure

## Installation

### Create Hugging Face Token Secret

Before installing the chart, create a Kubernetes secret with your Hugging Face token:

```bash
kubectl create secret generic hf-token --from-literal=token=your_huggingface_token
```

## Configuration

The following table lists the configurable parameters of the inference-charts chart and their default values.

| Parameter                                                                | Description                                                                         | Default                                                                     |
|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `global.image.pullPolicy`                                                | Global image pull policy                                                            | `IfNotPresent`                                                              |
| `inference.accelerator`                                                  | Accelerator type to use (gpu or neuron)                                             | `gpu`                                                                       |
| `inference.framework`                                                    | Framework type to use (vllm, ray-vllm, triton-vllm, aibrix, lws-vllm, or diffusers) | `vllm`                                                                      |
| `inference.serviceName`                                                  | Name of the inference service                                                       | `inference`                                                                 |
| `inference.serviceNamespace`                                             | Namespace for the inference service                                                 | `default`                                                                   |
| `inference.modelServer.image.repository`                                 | Model server image repository                                                       | `vllm/vllm-openai`                                                          |
| `inference.modelServer.image.tag`                                        | Model server image tag                                                              | `latest`                                                                    |
| `inference.modelServer.vllmVersion`                                      | VLLM version (for Ray deployments)                                                  | Not set                                                                     |
| `inference.modelServer.pythonVersion`                                    | Python version (for Ray deployments)                                                | Not set                                                                     |
| `inference.modelServer.env`                                              | Custom environment variables                                                        | `{}`                                                                        |
| `inference.modelServer.deployment.replicas`                              | Number of replicas                                                                  | `1`                                                                         |
| `inference.modelServer.deployment.minReplicas`                           | Minimum number of replicas (for Ray)                                                | `1`                                                                         |
| `inference.modelServer.deployment.maxReplicas`                           | Maximum number of replicas (for Ray)                                                | `2`                                                                         |
| `inference.modelServer.deployment.instanceType`                          | Node selector for instance type                                                     | Not set                                                                     |
| `inference.modelServer.deployment.topologySpreadConstraints.enabled`     | Enable topology spread constraints                                                  | `true`                                                                      |
| `inference.modelServer.deployment.topologySpreadConstraints.constraints` | List of topology spread constraints                                                 | See default configuration                                                   |
| `inference.modelServer.deployment.podAffinity.enabled`                   | Enable pod affinity                                                                 | `true`                                                                      |
| `inference.rayOptions.rayVersion`                                        | Ray version to use                                                                  | `2.47.0`                                                                    |
| `inference.rayOptions.autoscaling.enabled`                               | Enable Ray native autoscaling                                                       | `false`                                                                     |
| `inference.rayOptions.autoscaling.upscalingMode`                         | Ray autoscaler upscaling mode                                                       | `Default`                                                                   |
| `inference.rayOptions.autoscaling.idleTimeoutSeconds`                    | Idle timeout before scaling down                                                    | `60`                                                                        |
| `inference.rayOptions.autoscaling.actorAutoscaling.minActors`            | Minimum number of actors                                                            | `1`                                                                         |
| `inference.rayOptions.autoscaling.actorAutoscaling.maxActors`            | Maximum number of actors                                                            | `1`                                                                         |
| `inference.rayOptions.observability.rayPrometheusHost`                   | Ray Prometheus host URL                                                             | `http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090` |
| `inference.rayOptions.observability.rayGrafanaHost`                      | Ray Grafana host URL                                                                | `http://kube-prometheus-stack-grafana.monitoring.svc.cluster.local`         |
| `inference.rayOptions.observability.rayGrafanaIframeHost`                | Ray Grafana iframe host URL                                                         | `http://localhost:3000`                                                     |
| `vllm.logLevel`                                                          | Log level for VLLM                                                                  | `debug`                                                                     |
| `vllm.port`                                                              | VLLM server port                                                                    | `8004`                                                                      |
| `service.type`                                                           | Service type                                                                        | `ClusterIP`                                                                 |
| `service.port`                                                           | Service port                                                                        | `8000`                                                                      |
| `fluentbit.image.repository`                                             | Fluent Bit image repository                                                         | `fluent/fluent-bit`                                                         |
| `fluentbit.image.tag`                                                    | Fluent Bit image tag                                                                | `3.2.2`                                                                     |

### Model Parameters

The chart provides configuration for various model parameters:

| Parameter                                   | Description                           | Default                     |
|---------------------------------------------|---------------------------------------|-----------------------------|
| `model`                                     | Model ID from Hugging Face Hub        | `NousResearch/Llama-3.2-1B` |
| `modelParameters.gpuMemoryUtilization`      | GPU memory utilization                | `0.8`                       |
| `modelParameters.maxModelLen`               | Maximum model sequence length         | `8192`                      |
| `modelParameters.maxNumSeqs`                | Maximum number of sequences           | `4`                         |
| `modelParameters.maxNumBatchedTokens`       | Maximum number of batched tokens      | `8192`                      |
| `modelParameters.tokenizerPoolSize`         | Tokenizer pool size                   | `4`                         |
| `modelParameters.maxParallelLoadingWorkers` | Maximum parallel loading workers      | `2`                         |
| `modelParameters.pipelineParallelSize`      | Pipeline parallel size                | `1`                         |
| `modelParameters.tensorParallelSize`        | Tensor parallel size                  | `1`                         |
| `modelParameters.enablePrefixCaching`       | Enable prefix caching                 | `true`                      |
| `modelParameters.pipeline`                  | Pipeline type for diffusers framework | Not set                     |

**Note**: Model parameters are automatically converted to command line arguments in kebab-case format (e.g.,`maxNumSeqs`
becomes `--max-num-seqs`). For diffusers deployments, the `pipeline` parameter specifies the diffusion pipeline type to
use.

### Ray GCS High Availability Parameters

For Ray-VLLM deployments, you can enable GCS (Global Control Store) high availability:

| Parameter                                                           | Description                           | Default       |
|---------------------------------------------------------------------|---------------------------------------|---------------|
| `inference.rayOptions.gcs.highAvailability.enabled`                 | Enable GCS high availability          | `false`       |
| `inference.rayOptions.gcs.highAvailability.redis.address`           | Address for redis                     | `redis.redis` |
| `inference.rayOptions.gcs.highAvailability.redis.port`              | Port for redis                        | `6379`        |
| `inference.rayOptions.gcs.highAvailability.redis.secretName`        | Secret name containing redis password | ``            |
| `inference.rayOptions.gcs.highAvailability.redis.secretPasswordKey` | Key in secret with redis password     | ``            |

## Supported Models

The chart includes pre-configured values files for the following models:

### GPU Models

#### Language Models

- **DeepSeek R1 Distill Llama 8B**: `values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml` (Ray-VLLM)
- **Llama 3.2 1B**: `values-llama-32-1b-vllm.yaml` (VLLM), `values-llama-32-1b-ray-vllm.yaml` (Ray-VLLM),
  `values-llama-32-1b-ray-vllm-autoscaling.yaml` (Ray-VLLM with autoscaling),
  `values-llama-32-1b-aibrix.yaml` (AIBrix), and `values-llama-32-1b-triton-vllm-gpu.yaml` (Triton-VLLM)
- **Llama 4 Scout 17B**: `values-llama-4-scout-17b-vllm.yaml` (VLLM) and `values-llama-4-scout-17b-lws-vllm.yaml` (
  LeaderWorkerSet-VLLM)
- **Mistral Small 24B**: `values-mistral-small-24b-ray-vllm.yaml` (Ray-VLLM)
- **GPT OSS 20B**: `values-gpt-oss-20b-vllm.yaml` (VLLM)
- **Qwen 3 Coder 480B** `values-qwen-3-coder-480b-a35b-instruct-lws-vllm.yaml`(LeaderWorkerSet-VLLM)

#### Diffusion Models

- **FLUX.1 Schnell**: `values-flux-1-diffusers.yaml` (Diffusers)
- **Kolors**: `values-kolors-diffusers.yaml` (Diffusers)
- **Stable Diffusion 3.5 Large**: `values-stable-diffusion-3.5-large-diffusers.yaml` (Diffusers)
- **Stable Diffusion XL Base 1.0**: `values-stable-diffusion-xl-base-1-diffusers.yaml` (Diffusers)
- **Latent Diffusion**: `values-latent-diffusion-diffusers.yaml` (Diffusers)
- **OmniGen**: `values-omni-gen-diffusers.yaml` (Diffusers)

### Neuron Models

- **DeepSeek R1 Distill Llama 8B**: `values-deepseek-r1-distill-llama-8b-vllm-neuron.yaml` (VLLM)
- **Llama 2 13B**: `values-llama-2-13b-ray-vllm-neuron.yaml` (Ray-VLLM)
- **Llama 3 70B**: `values-llama-3-70b-ray-vllm-neuron.yaml` (Ray-VLLM)
- **Llama 3.1 8B**: `values-llama-31-8b-vllm-neuron.yaml` (VLLM) and `values-llama-31-8b-ray-vllm-neuron.yaml` (
  Ray-VLLM)

## Topology Spread Constraints

The chart includes optional topology spread constraints to control how pods are distributed across your cluster. By
default, the chart is configured to prefer scheduling replicas in the same availability zone for reduced network latency
and cost optimization.

### Default Configuration

```yaml
inference:
  modelServer:
    deployment:
      topologySpreadConstraints:
        enabled: true
        constraints:
          # Prefer same AZ as head pod (soft constraint)
          - maxSkew: 1
            topologyKey: topology.kubernetes.io/zone
            whenUnsatisfiable: ScheduleAnyway
            labelSelector:
              matchLabels: { }
          # Require workers to be grouped together (hard constraint)
          - maxSkew: 1
            topologyKey: topology.kubernetes.io/zone
            whenUnsatisfiable: DoNotSchedule
            labelSelector:
              matchLabels: { }
      podAffinity:
        enabled: true
        # Strong preference for same AZ (helps Karpenter understand intent)
        preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              topologyKey: topology.kubernetes.io/zone
              labelSelector:
                matchLabels: { }
```

**Note**: For Ray deployments, the default configuration uses two constraints:

1. **Head Co-location**: Workers prefer to be in the same AZ as the head pod (soft constraint)
2. **Worker Grouping**: All worker pods must be scheduled together in the same AZ (hard constraint)

This ensures optimal performance while maintaining high availability - workers will try to co-locate with the head, but
if that's not possible, they'll at least be grouped together for consistent inter-worker communication.

### Disabling Topology Constraints

To disable topology spread constraints entirely:

```yaml
inference:
  modelServer:
    deployment:
      topologySpreadConstraints:
        enabled: false
      podAffinity:
        enabled: false
```

### Ray-Specific Behavior

For Ray deployments (`framework: ray-vllm`), the topology constraints work differently:

- **Head Group**: Uses the first constraint to establish zone preference
- **Worker Group**: Uses both constraints:
    1. First constraint (soft): Tries to co-locate with head pod
    2. Second constraint (hard): Ensures all workers are grouped together

**Scheduling Logic**:

1. Head pod schedules in any available zone
2. Workers try to schedule in the same zone as head
3. If head's zone is full, workers schedule together in another zone
4. Workers are never split across multiple zones

### Karpenter Compatibility

The chart uses **both topology spread constraints and pod affinity** for Karpenter compatibility:

- **Topology Spread Constraints**: Control pod distribution at the scheduler level
- **Pod Affinity**: Help Karpenter understand co-location intent during node provisioning

**Troubleshooting Steps**:

1. **Soft constraints first**: Always start with `whenUnsatisfiable: ScheduleAnyway`
2. **Check node availability**: Verify nodes exist in your target AZ
3. **Monitor Karpenter logs**: Check why it's provisioning in different AZs

## Ray Native Autoscaling

For Ray-VLLM deployments, you can enable Ray's native autoscaling feature which automatically scales worker nodes based
on workload demand. This is more efficient than Kubernetes HPA as it understands Ray's internal workload distribution.

### Autoscaling Configuration

| Parameter                                                     | Description                                     | Default   |
|---------------------------------------------------------------|-------------------------------------------------|-----------|
| `inference.rayOptions.autoscaling.enabled`                    | Enable Ray native autoscaling                   | `false`   |
| `inference.rayOptions.autoscaling.upscalingMode`              | Ray autoscaler upscaling mode                   | `Default` |
| `inference.rayOptions.autoscaling.idleTimeoutSeconds`         | How long to wait before scaling down idle nodes | `60`      |
| `inference.rayOptions.autoscaling.actorAutoscaling.minActors` | Minimum number of actors                        | `1`       |
| `inference.rayOptions.autoscaling.actorAutoscaling.maxActors` | Maximum number of actors                        | `1`       |

### Example Autoscaling Configuration

```yaml
inference:
  framework: ray-vllm
  rayOptions:
    autoscaling:
      enabled: true
      upscalingMode: "Aggressive"
      idleTimeoutSeconds: 120  # Wait 2 minutes before scaling down
      actorAutoscaling:
        minActors: 1
        maxActors: 5
```

## Diffusers Framework

The Diffusers framework provides support for Hugging Face Diffusers library, enabling deployment of various image
generation and diffusion models on GPU infrastructure.

### Supported Pipeline Types

The chart supports multiple diffusion pipeline types through the `modelParameters.pipeline` configuration:

| Pipeline Type      | Description                                                     | Example Models                        |
|--------------------|-----------------------------------------------------------------|---------------------------------------|
| `flux`             | Standard Stable Diffusion pipeline for text-to-image generation | FLUX.1-schnell                        |
| `diffusion`        | Generic diffusion pipeline for various diffusion models         | Stable Diffusion XL, Latent Diffusion |
| `kolors`           | Kolors-specific pipeline for Kolors diffusion models            | Kwai-Kolors/Kolors-diffusers          |
| `stablediffusion3` | Stable Diffusion 3.x pipeline with enhanced capabilities        | Stable Diffusion 3.5 Large            |
| `omnigen`          | OmniGen pipeline for multi-modal generation                     | Shitao/OmniGen-v1                     |

### Diffusers Configuration

For Diffusers deployments, use the following configuration structure:

```yaml
model: stabilityai/stable-diffusion-xl-base-1.0

modelParameters:
  pipeline: diffusion

inference:
  serviceName: sd-diffusers
  serviceNamespace: default
  accelerator: gpu
  framework: diffusers

  modelServer:
    image:
      repository: diffusers/diffusers-pytorch-cuda
      tag: latest
    deployment:
      instanceType: g6e.2xlarge
```

### Hardware Requirements

Diffusers deployments are optimized for GPU inference and typically require:

- **GPU Memory**: 8GB+ VRAM recommended for most models
- **Instance Types**: g6e.2xlarge or higher recommended
- **Storage**: Sufficient space for model weights (varies by model, typically 2-10GB)

### API Endpoints

Diffusers deployments expose REST API endpoints for image generation:

- `/v1/generations` - Primary image generation endpoint

### Example API Usage

```bash
# Generate an image using the diffusers API
curl -X POST http://localhost:8000/v1/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A beautiful sunset over mountains",
  }'
```

## Examples

### Deploy GPU Ray-VLLM with DeepSeek R1 Distill Llama 8B model

```bash
helm install deepseek-gpu-inference ./inference-charts --values values-deepseek-r1-distill-llama-8b-ray-vllm-gpu.yaml
```

### Deploy GPU VLLM with Llama 3.2 1B model

```bash
helm install gpu-vllm-inference ./inference-charts --values values-llama-32-1b-vllm.yaml
```

### Deploy GPU LeaderWorkerSet-VLLM with Llama 4 Scout 17B model

```bash
helm install llama4-lws-inference ./inference-charts --values values-llama-4-scout-17b-lws-vllm.yaml
```

### Deploy GPU LeaderWorkerSet-VLLM with Qwen3 Coder 480B A35B Instruct model

```bash
helm install qwen3-coder-lws-inference ./inference-charts --values values-qwen-3-coder-480b-a35b-instruct-lws-vllm.yaml
```

### Deploy GPU Ray-VLLM with Llama 3.2 1B model

```bash
helm install gpu-ray-vllm-inference ./inference-charts --values values-llama-32-1b-ray-vllm.yaml
```

### Deploy GPU AIBrix with Llama 3.2 1B model

```bash
helm install gpu-aibrix-inference ./inference-charts --values values-llama-32-1b-aibrix.yaml
```

### Deploy Neuron VLLM with DeepSeek R1 Distill Llama 8B model

```bash
helm install deepseek-neuron-inference ./inference-charts --values values-deepseek-r1-distill-llama-8b-vllm-neuron.yaml
```

### Deploy Neuron Ray-VLLM with Llama 2 13B model

```bash
helm install llama2-neuron-inference ./inference-charts --values values-llama-2-13b-ray-vllm-neuron.yaml
```

### Deploy Neuron Ray-VLLM with Llama 3 70B model

```bash
helm install llama3-70b-neuron-inference ./inference-charts --values values-llama-3-70b-ray-vllm-neuron.yaml
```

### Deploy Neuron VLLM with Llama 3.1 8B model

```bash
helm install neuron-vllm-inference ./inference-charts --values values-llama-31-8b-vllm-neuron.yaml
```

### Deploy Neuron Ray-VLLM with Llama 3.1 8B model

```bash
helm install neuron-ray-vllm-inference ./inference-charts --values values-llama-31-8b-ray-vllm-neuron.yaml
```

### Deploy GPU Ray-VLLM with Mistral Small 24B model

```bash
helm install gpu-ray-vllm-mistral ./inference-charts --values values-mistral-small-24b-ray-vllm.yaml
```

### Deploy GPU Ray-VLLM with Llama 3.2 1B model with autoscaling

```bash
helm install gpu-ray-vllm-autoscale ./inference-charts --values values-llama-32-1b-ray-vllm-autoscaling.yaml
```

### Deploy GPU Triton-VLLM with Llama 3.2 1B

```bash
helm install gpu-triton-vllm ./inference-charts --values values-llama-32-1b-triton-vllm-gpu.yaml
```

### Deploy GPT OSS 20B with VLLM

```bash
helm install gpt-oss-vllm ./inference-charts --values values-gpt-oss-20b-vllm.yaml
```

### Deploy Diffusers Models

#### Deploy FLUX.1 Schnell for image generation

```bash
helm install flux-diffusers ./inference-charts --values values-flux-1-diffusers.yaml
```

#### Deploy Stable Diffusion XL Base 1.0

```bash
helm install sdxl-diffusers ./inference-charts --values values-stable-diffusion-xl-base-1-diffusers.yaml
```

#### Deploy Stable Diffusion 3.5 Large

```bash
helm install sd3-diffusers ./inference-charts --values values-stable-diffusion-3.5-large-diffusers.yaml
```

#### Deploy Kolors for artistic image generation

```bash
helm install kolors-diffusers ./inference-charts --values values-kolors-diffusers.yaml
```

#### Deploy OmniGen for multi-modal generation

```bash
helm install omnigen-diffusers ./inference-charts --values values-omni-gen-diffusers.yaml
```

#### Deploy Latent Diffusion

```bash
helm install latent-diffusion ./inference-charts --values values-latent-diffusion-diffusers.yaml
```

### Custom Deployment

You can also create your own values file with custom settings:

```yaml
inference:
  accelerator: gpu  # or neuron
  framework: vllm   # or ray-vllm, triton-vllm, aibrix, lws-vllm, or diffusers
  serviceName: custom-inference
  serviceNamespace: default

  # Ray-specific options (only for ray-vllm framework)
  rayOptions:
    rayVersion: 2.47.0
    autoscaling:
      enabled: false
      upscalingMode: "Default"
      idleTimeoutSeconds: 60
      actorAutoscaling:
        minActors: 1
        maxActors: 1
    observability:
      rayPrometheusHost: http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090
      rayGrafanaHost: http://kube-prometheus-stack-grafana.monitoring.svc.cluster.local
      rayGrafanaIframeHost: http://localhost:3000

  modelServer:
    # For Ray deployments, specify VLLM and Python versions
    vllmVersion: 0.9.1
    pythonVersion: 3.11
    image:
      repository: vllm/vllm-openai  # Use rayproject/ray for Ray deployments
      tag: latest
    deployment:
      replicas: 1
      minReplicas: 1
      maxReplicas: 2
      resources:
        gpu:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
    env: { }  # Custom environment variables

model: "NousResearch/Llama-3.2-1B"

modelParameters:
  gpuMemoryUtilization: 0.8
  maxModelLen: 8192
  maxNumSeqs: 4
  maxNumBatchedTokens: 8192
  tokenizerPoolSize: 4
  maxParallelLoadingWorkers: 2
  pipelineParallelSize: 1
  tensorParallelSize: 1
  enablePrefixCaching: true

# For diffusers deployments, use this configuration instead:
# model: "stabilityai/stable-diffusion-xl-base-1.0"
# modelParameters:
#   pipeline: diffusion
```

Then install the chart with your custom values:

```bash
helm install custom-inference ./inference-charts --values custom-values.yaml
```

## API Endpoints

### VLLM and Ray-VLLM Deployments

The deployed service exposes the following OpenAI-compatible API endpoints:

- `/v1/models` - List available models
- `/v1/completions` - Text completion API
- `/v1/chat/completions` - Chat completion API
- `/metrics` - Prometheus metrics endpoint

### Triton-VLLM Deployments

The deployed service exposes the following Triton Inference Server API endpoints:

**HTTP API (Port 8000):**

- `/v2/health/live` - Liveness check
- `/v2/health/ready` - Readiness check
- `/v2/models` - List available models
- `/v2/models/vllm_model/generate` - Model inference endpoint

**gRPC API (Port 8001):**

- Standard Triton gRPC inference protocol

**Metrics (Port 8002):**

- `/metrics` - Prometheus metrics endpoint

**Example Triton API Usage:**

```bash
# Check model status
curl http://localhost:8000/v2/models/llama-3-2-1b

# Run inference
curl -X POST http://localhost:8000/v2/models/vllm_model/generate \
  -H 'Content-Type: application/json' \
  -d '{"text_input":"what is the capital of France?"}'
```

### Diffusers Deployments

The deployed service exposes REST API endpoints for image generation:

- `/v1/generations` - Primary image generation endpoint

**Example Diffusers API Usage:**

```bash
# Generate an image using the diffusers API
curl -X POST http://localhost:8000/v1/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A beautiful sunset over mountains"
  }'
```

## Ray GCS High Availability

For production Ray-VLLM deployments, you can enable GCS (Global Control Store) high availability using the RayService
CRD's native support to ensure fault tolerance and prevent single points of failure.

### Features

- **Native CRD Support**: Uses RayService CRD's built-in GCS HA configuration
- **Fault Tolerance**: GCS state is persisted to Redis, allowing recovery from head node failures
- **Automatic Recovery**: Ray cluster can recover from GCS failures without losing job state
- **Scalability**: Multiple GCS replicas can handle increased load
- **Flexible Storage**: Support for both internal Redis (deployed with the chart) and external Redis clusters

### Example Configuration

The chart uses the RayService CRD's native GCS HA configuration:

Create a secret for the redis password if needed (replace REDISPASSWORD with your password)

```bash
kubectl create secret generic redis-secret --from-literal=redis-password=REDISPASSWORD
```

```yaml
inference:
  framework: ray-vllm
  rayOptions:
    gcs:
      highAvailability:
        enabled: true
        redis:
          address: redis.redis # redis service in redis namespace
          port: 6379
          secretName: redis-secret
          secretPasswordKey: redis-password
```

## Troubleshooting GCS High Availability

### Common Issues

1. **Redis Connection Issues**
    - Check Redis service is running: `kubectl get pods -l app.kubernetes.io/component=redis-gcs`
    - Verify Redis connectivity: `kubectl exec -it <ray-head-pod> -- redis-cli -h <redis-service> ping`

2. **GCS Recovery**
    - Check RayService status: `kubectl get rayservice <service-name> -o yaml`
    - Check GCS logs: `kubectl logs <ray-head-pod> -c head | grep -i gcs`
    - Verify Redis contains GCS state: `kubectl exec -it <redis-pod> -- redis-cli keys "*"`

3. **Performance Issues**
    - Increase Redis resources if experiencing timeouts
    - Monitor Redis memory usage

### Monitoring

- GCS metrics are available at `/metrics` endpoint
- Redis metrics can be monitored using Redis Exporter
- Ray dashboard shows cluster health and GCS status

## Observability

The chart includes Fluent Bit for log collection and exposes Prometheus metrics for monitoring. The Ray-VLLM deployment
also includes configuration for Grafana dashboards.
