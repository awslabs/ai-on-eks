---
title: NVIDIA NIM Operator on EKS
sidebar_position: 4
---
import CollapsibleContent from '../../../../src/components/CollapsibleContent';

:::warning

Note: Before implementing NVIDIA NIM, please be aware it is part of [NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/), which may introduce potential cost and licensing for production use.

For evaluation, NVIDIA also offers a free evaluation license to try NVIDIA AI Enterprise for 90 days, and you can [register](https://enterpriseproductregistration.nvidia.com/?LicType=EVAL&ProductFamily=NVAIEnterprise) it with your corporate email.
:::


# NVIDIA NIM Operator on Amazon EKS

## [What is NVIDIA NIM?](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html)

**NVIDIA NIM** ([NVIDIA Inference Microservices](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html)) is a set of containerized microservices that make it easier to deploy and host large language models (LLMs) and other AI models in your own environment. NIM provides standard APIs (similar to OpenAI or other AI services) for developers to build applications like chatbots and AI assistants, while leveraging NVIDIA’s GPU acceleration for high-performance inference. In essence, NIM abstracts away the complexities of model runtime and optimization, offering a fast path to inference with optimized backends (e.g., TensorRT-LLM, FasterTransformer, etc.) under the hood.

## [NVIDIA NIM Operator for Kubernetes](https://docs.nvidia.com/nim-operator/latest/index.html#)

The **NVIDIA NIM Operator** is a Kubernetes operator that automates the deployment, scaling, and management of NVIDIA NIM microservices on a Kubernetes cluster.

Instead of manually pulling containers, provisioning GPU nodes, or writing YAML for every model, the NIM Operator introduces two primary [Custom Resource Definitions (CRDs)](https://docs.nvidia.com/nim-operator/latest/crds.html):
- [`NIMCache`](https://docs.nvidia.com/nim-operator/latest/cache.html)
- [`NIMService`](https://docs.nvidia.com/nim-operator/latest/service.html)

These CRDs allow you to declaratively define model deployments using native Kubernetes syntax.

The Operator handles:
- Pulling the model image from NVIDIA GPU Cloud (NGC)
- Caching model weights and optimized runtime profiles
- Launching model-serving pods with GPU allocation
- Exposing inference endpoints via Kubernetes Services
- Integrating with autoscaling (e.g., HPA + Karpenter)

This blueprint focuses on **inference**, and the two key CRDs we use are explained below:


### [NIMCache – Model Caching for Faster Load Times](https://docs.nvidia.com/nim-operator/latest/cache.html)

A `NIMCache` (`nimcaches.apps.nvidia.com`) is a custom resource that pre-downloads and stores a model’s weights, tokenizer, and runtime-optimized engine files (such as TensorRT-LLM profiles) into a shared persistent volume.

This ensures:
- **Faster cold start times**: no repeated downloads from NGC
- **Storage reuse across nodes and replicas**
- **Centralized, shared model store** (typically on EFS or FSx for Lustre in EKS)

Model profiles are optimized for specific GPUs (e.g., A10G, L4) and precisions (e.g., FP16). When NIMCache is created, the Operator discovers available model profiles and selects the best one for your cluster.

> 📌 Tip: Using `NIMCache` is highly recommended for production, especially when running multiple replicas or restarting models frequently.

### [NIMService – Deploying and Managing the Model Server](https://docs.nvidia.com/nim-operator/latest/service.html)

A `NIMService` (`nimservices.apps.nvidia.com`) represents a running instance of a NIM model server on your cluster. It specifies the container image, GPU resources, number of replicas, and optionally the name of a `NIMCache`.

Key benefits:
- **Declarative model deployment** using Kubernetes YAML
- **Automatic node scheduling** for GPU nodes
- **Shared cache support** using `NIMCache`
- **Autoscaling** via HPA or external triggers
- **ClusterIP or Ingress support** to expose APIs

For example, deploying the Meta Llama 3.1 8B Instruct model involves creating:
- A `NIMCache` to store the model (optional but recommended)
- A `NIMService` pointing to the cached model and allocating GPUs

If `NIMCache` is not used, the model will be downloaded each time a pod starts, which may increase startup latency.

### [NIMPipeline](https://docs.nvidia.com/nim-operator/latest/pipeline.html)

`NIMPipeline` is another CRD that can group multiple `NIMService` resources into an ordered inference pipeline. This is useful for multi-model workflows like:
- Retrieval-Augmented Generation (RAG)
- Embeddings + LLM chaining
- Preprocessing + classification pipelines

> In this tutorial, we focus on a single model deployment using `NIMCache` and `NIMService`.

## Overview of this deployment pattern on Amazon EKS

This deployment blueprint demonstrates how to run the **Meta Llama 3.1 8B Instruct** model on **Amazon EKS** using the **NVIDIA NIM Operator** with multi-GPU support and optimized model caching for fast startup times.

The model is served using:
- **G5 instances (g5.12xlarge)**: These instances come with **4 NVIDIA A10G GPUs**
- **Tensor Parallelism (TP)**: Set to `2`, meaning the model will run in parallel across **2 GPUs**
- **Persistent Shared Cache**: Backed by Amazon **EFS** to speed up model startup by reusing previously generated engine files

By combining these components, the model is deployed as a scalable Kubernetes workload that supports:
- Efficient GPU scheduling with [Karpenter](https://karpenter.sh/)
- Fast model load using the [`NIMCache`](https://docs.nvidia.com/nim-operator/latest/cache.html)
- Scalable serving endpoint via [`NIMService`](https://docs.nvidia.com/nim-operator/latest/service.html)

> 📌 Note: You can modify the `tensorParallelism` setting or select a different instance type (e.g., G6 with L4 GPUs) based on your performance and cost requirements.

## Deploying the Solution

In this tutorial, the entire AWS infrastructure is provisioned using **Terraform**, including:
- Amazon VPC with public and private subnets
- Amazon EKS cluster
- GPU nodepools using **Karpenter**
- Addons such as:
  - NVIDIA device plugin
  - EFS CSI driver
  - **NVIDIA NIM Operator**

As a demonstration, the **Meta Llama-3.1 8B Instruct** model will be deployed using a `NIMService`, optionally backed by a `NIMCache` for improved cold start performance.

### Prerequisites

Before getting started with NVIDIA NIM, ensure you have the following:

<details>
<summary>Click to expand the NVIDIA NIM account setup details</summary>

**NVIDIA AI Enterprise Account**

- Register for an NVIDIA AI Enterprise account. If you don't have one, you can sign up for a trial account using this [link](https://enterpriseproductregistration.nvidia.com/?LicType=EVAL&ProductFamily=NVAIEnterprise).

**NGC API Key**

1. Log in to your NVIDIA AI Enterprise account
2. Navigate to the NGC (NVIDIA GPU Cloud) [portal](https://org.ngc.nvidia.com/)
3. Generate a personal API key:
    - Go to your account settings or navigate directly to: https://org.ngc.nvidia.com/setup/personal-keys
    - Click on "Generate Personal Key"
    - Ensure that at least "NGC Catalog" is selected from the "Services Included" dropdown
    - Copy and securely store your API key, the key should have a prefix with `nvapi-`

    ![NGC API KEY](../img/nim-ngc-api-key.png)

**Validate NGC API Key and Test Image Pull**

To ensure your API key is valid and working correctly:
1. Set up your NGC API key as an environment variable:
```bash
export NGC_API_KEY=<your_api_key_here>
```

2. Authenticate Docker with the NVIDIA Container Registry:

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

3. Test pulling an image from NGC:
```bash
docker pull nvcr.io/nim/meta/llama3-8b-instruct:latest
```
You do not have to wait for it to complete, just to make sure the API key is valid to pull the image.
</details>

The following are required to run this tutorial
- An active AWS account with admin equivalent permissions
- [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
- [kubectl](https://Kubernetes.io/docs/tasks/tools/)
- [Terraform](https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli)

### Deploy

Clone the [ai-on-eks](https://github.com/awslabs/ai-on-eks) repository that contains the Terraform code for this deployment pattern:

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```
Navigate to the NVIDIA NIM deployment directory and run the install script to deploy the infrastructure:

```bash
cd ai-on-eks/infra/nvidia-nim
./install.sh
```

This deployment will take approximately `~20 minute` to complete.

Once the installation finishes, you may find the `configure_kubectl` command from the output. Run the following to configure EKS cluster access

```bash
# Creates k8s config file to authenticate with EKS
aws eks --region us-west-2 update-kubeconfig --name nvidia-nim-eks
```

<details>
<summary>Verify the deployments - Click to expand the deployment details</summary>

$ kubectl get all -n nim-operator

```
kubectl get all -n nim-operator
NAME                                                 READY   STATUS    RESTARTS   AGE
pod/nim-operator-k8s-nim-operator-6fdffdf97f-56fxc   1/1     Running   0          26h

NAME                                       TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)    AGE
service/k8s-nim-operator-metrics-service   ClusterIP   172.20.148.6   <none>        8080/TCP   26h

NAME                                            READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/nim-operator-k8s-nim-operator   1/1     1            1           26h

NAME                                                       DESIRED   CURRENT   READY   AGE
replicaset.apps/nim-operator-k8s-nim-operator-6fdffdf97f   1         1         1       26h
```

$ kubectl get crds | grep nim

```
nimcaches.apps.nvidia.com                    2025-03-27T17:39:00Z
nimpipelines.apps.nvidia.com                 2025-03-27T17:39:00Z
nimservices.apps.nvidia.com                  2025-03-27T17:39:01Z
```

$ kubectl get crds | grep nemo

```
nemocustomizers.apps.nvidia.com              2025-03-27T17:38:59Z
nemodatastores.apps.nvidia.com               2025-03-27T17:38:59Z
nemoentitystores.apps.nvidia.com             2025-03-27T17:38:59Z
nemoevaluators.apps.nvidia.com               2025-03-27T17:39:00Z
nemoguardrails.apps.nvidia.com               2025-03-27T17:39:00Z
```

To list Karpenter autoscaling Nodepools

$ kubectl get nodepools

```
NAME                NODECLASS           NODES   READY   AGE
g5-gpu-karpenter    g5-gpu-karpenter    1       True    47h
g6-gpu-karpenter    g6-gpu-karpenter    0       True    7h56m
inferentia-inf2     inferentia-inf2     0       False   47h
trainium-trn1       trainium-trn1       0       False   47h
x86-cpu-karpenter   x86-cpu-karpenter   0       True    47h
```

</details>


### Deploy llama-3.1-8b-instruct with NIM Operator

#### Step 1: Create Secrets for Authentication
To access the NVIDIA container registry and model artifacts, you'll need to provide your NGC API key. This script creates two Kubernetes secrets: `ngc-secret` for Docker image pulls and `ngc-api-secret` for model authorization.

```bash
cd blueprints/inference/gpu/nvidia-nim-operator-llama3-8b

NGC_API_KEY="your-real-ngc-key" ./deploy-nim-auth.sh
```

#### Step 2: Cache the Model to EFS using NIMCache CRD

The `NIMCache` custom resource will pull the model and cache optimized engine profiles to EFS. This dramatically reduces startup time when launching the model later via `NIMService`.

```bash
kubectl apply -f nim-cache-llama3-8b-instruct.yaml
```

Check status:

```bash
kubectl get nimcaches.apps.nvidia.com -n nim-service
```

Expected output:

```
NAME                      STATUS   PVC                           AGE
meta-llama3-8b-instruct   Ready    meta-llama3-8b-instruct-pvc   21h
```


Display cached model profiles:

```bash
kubectl get nimcaches.apps.nvidia.com -n nim-service \
  meta-llama3-8b-instruct -o=jsonpath="{.status.profiles}" | jq .
```

Sample output:

```json
[
  {
    "config": {
      "feat_lora": "false",
      "gpu": "A10G",
      "llm_engine": "tensorrt_llm",
      "precision": "fp16",
      "profile": "throughput",
      "tp": "2"
    }
  }
]
```

#### Step 3: Deploy the Model using NIMService CRD

Now launch the model service using the cached engine profiles.

```bash
kubectl apply -f nim-service-llama3-8b-instruct.yaml
```

Check the deployed resources:

```bash
kubectl get all -n nim-service
```

Exepcted Output:

```
NAME                                           READY   STATUS    RESTARTS   AGE
pod/meta-llama3-8b-instruct-6cdf47d6f6-hlbnf   1/1     Running   0          6h35m

NAME                              TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)    AGE
service/meta-llama3-8b-instruct   ClusterIP   172.20.85.8   <none>        8000/TCP   6h35m

NAME                                      READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/meta-llama3-8b-instruct   1/1     1            1           6h35m

NAME                                                 DESIRED   CURRENT   READY   AGE
replicaset.apps/meta-llama3-8b-instruct-6cdf47d6f6   1         1         1       6h35m
```

### 🚀 Model Startup Timeline

The following sample is captured from the `pod/meta-llama3-8b-instruct-6cdf47d6f6-hlbnf` log

| Step              | Timestamp       | Description                                                                 |
|-------------------|-----------------|-----------------------------------------------------------------------------|
| Start             | ~20:00:50       | Pod starts, NIM container logs begin                                       |
| Profile Match     | 20:00:50.100    | Detects and selects cached profile (tp=2)                                  |
| Workspace Ready   | 20:00:50.132    | Model workspace initialized via EFS in 0.126s                              |
| TensorRT Init     | 20:00:51.168    | TensorRT-LLM engine begins setup                                           |
| Engine Ready      | 20:01:06        | Engine loaded and profiles activated (~16.6 GiB across 2 GPUs)             |
| API Server Ready  | 20:02:11.036    | FastAPI + Uvicorn starts                                                   |
| Health Check OK   | 20:02:18.781    | `/v1/health/ready` endpoint returns 200 OK                                 |

> ⚡ **Startup time (cold boot to ready): ~81 seconds** thanks to cached engine on EFS.

### Test the Model with a Prompt

#### Step 1: Port Forward the Model Service

Expose the model locally using port forwarding:

```bash
kubectl port-forward -n nim-service service/meta-llama3-8b-instruct 8001:8000
```

#### Step 2: Send a Sample Prompt Using curl

Run the following command to test the model with a chat prompt:

```sh
curl -X POST \
  http://localhost:8001/v1/chat/completions \
  -H 'Accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [
      {
        "role": "user",
        "content": "What should I do for a 4 day vacation at Cape Hatteras National Seashore?"
      }
    ],
    "top_p": 1,
    "n": 1,
    "max_tokens": 1024,
    "stream": false,
    "frequency_penalty": 0.0,
    "stop": ["STOP"]
  }'
```

**Sample Response (Shortened):**

```
{"id":"chat-061a9dba9179437fa24cab7f7c767f19","object":"chat.completion","created":1743215809,"model":"meta/llama-3.1-8b-instruct","choices":[{"index":0,"message":{"role":"assistant","content":"Cape Hatteras National Seashore is a beautiful coastal destination with a rich history, pristine beaches,
...
exploration of the area's natural beauty and history. Feel free to modify it to suit your interests and preferences. Safe travels!"},"logprobs":null,"finish_reason":"stop","stop_reason":null}],"usage":{"prompt_tokens":30,"total_tokens":773,"completion_tokens":743},"prompt_logprobs":null}%  

```


>🧠 The model is now running with Tensor Parallelism = 2 across two A10G GPUs, each utilizing approximately 21.4 GiB of memory. Thanks to NIMCache backed by EFS, the model loaded quickly and is ready for low-latency inference.

## Conclusion

This blueprint showcases how to deploy and scale **large language models like Meta’s Llama 3.1 8B Instruct** efficiently on **Amazon EKS** using the **NVIDIA NIM Operator**.

By combining **OpenAI-compatible APIs** with **GPU-accelerated inference**, **declarative Kubernetes CRDs** (`NIMCache`, `NIMService`), and **fast model startup via EFS-based caching**, you get a streamlined, production-grade model deployment experience.

### Key Benefits:
- **Faster cold starts** through shared, persistent model cache
- **Declarative and repeatable deployments** with CRDs
- **Dynamic GPU autoscaling** powered by Karpenter
- **One-click infrastructure provisioning** using Terraform

In just **~20 minutes**, you can go from zero to a **scalable LLM service on Kubernetes** — ready to serve real-world prompts with low latency and high efficiency.


## Cleanup

To tear down the deployed model and associated infrastructure:

### Step 1: Delete Model Resources

Delete the deployed `NIMService` and `NIMCache` objects from your cluster:

```bash
kubectl delete -f nim-service-llama3-8b-instruct.yaml
kubectl delete -f nim-cache-llama3-8b-instruct.yaml
```

**Verify deletion:**

```
kubectl get nimservices.apps.nvidia.com -n nim-service
kubectl get nimcaches.apps.nvidia.com -n nim-service
```

### Step 2: Destroy AWS Infrastructure

Navigate back to the root Terraform module and run the cleanup script. This will destroy all AWS resources created for this blueprint, including the VPC, EKS cluster, EFS, and node groups:

```bash
cd ai-on-eks/infra/nvidia-nim
./cleanup.sh
```


