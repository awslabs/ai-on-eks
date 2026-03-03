---
sidebar_label: Envoy Gateway implementation on EKS
---
import CollapsibleContent from '@site/src/components/CollapsibleContent';

# Envoy gateway

Organizations deploying AI applications face a fundamental challenge: no single model serves all needs. Developers may choose Claude for long-context analysis, OpenAI for reasoning tasks, and DeepSeek for cost-sensitive workloads. The problem is that each model provider uses different APIs. Without centralized control, teams can't easily switch providers, get visibility into utilization, or enforce quotas.

[Envoy AI Gateway](https://aigateway.envoyproxy.io/) is an open source project that solves this challenge by providing a single, scalable OpenAI-compatible endpoint that routes to multiple supported LLM providers. It gives Platform teams cost controls and observability, while developers never touch provider-specific SDKs.

## Key objectives of Envoy AI Gateway

- Provide a unified layer for routing and managing LLM/AI traffic
- Support automative failover mechanisms to ensure service reliability
- Ensure end-to-end security, including upstream authorization for LLM/AI traffic
- Implement a policy framework to support usage limiting use cases
- Foster an open-source community to address GenAI-specific routing and quality of service needs

## Envoy Gateway Fundamentals

:::info If you're already familiar with Envoy Gateway, you can skip this section.
:::

As Envoy AI Gateway builds on top of the standard Kubernetes Gateway API and Envoy Gateway extensions, it's necessary to familiarize yourself with the underlying Envoy Gateway primitives:

- **GatewayClass** - Defines which controller manages the Gateway. Envoy AI Gateway uses the same GatewayClass as Envoy Gateway.
- **Gateway** - The entry point for traffic. A Gateway resource defines listeners (HTTP/HTTPS ports). When you create a Gateway, Envoy Gateway deploys the actual Envoy proxy pods and a corresponding Kubernetes Service (typically a LoadBalancer). This is like a Network Load Balancer (although technically you'd still need to attach an NLB to an Envoy Gateway to accept traffic that's external the Kubernetes cluster.)
- **HTTPRoute** - The instruction for routing traffic HTTP based on hostnames, paths, or headers. Conceptually, this is similar to ingress rules or listener rules in ALB.
- **Backend** - A Kubernetes Service or an external endpoint.
- **BackendTrafficPolicy** - Configures connection behavior like timeouts, retries, and rate limiting of an HTTPRoute.
- **ClientTrafficPolicy** - Configures how the Envoy proxy server behaves with downstream clients.
- **EnvoyExtensionPolicy** - A way to extend Envoy's traffic processing capabilities.

Envoy AI Gateway introduces following CRDs:

- **AIGatewayRoute** - Defines unified API and routing rules for AI traffic
- **AIServiceBackend** - Represents individual AI service backends like Bedrock
- **BackendSecurityPolicy** - Configures authentication for backend access
- **BackendTLSPolicy** - Defines TLS parameters for backend connections

![envoy.png](./img/envoy.png)

This envoy gateway blueprint deploys Envoy AI Gateway on Amazon EKS and supports two use cases:

- Multi-model routing
- Rate limiting

<CollapsibleContent header={<h2><span>Deploying the Solution</span></h2>}>

We are utilizing Terraform Infrastructure as Code (IaC) templates to deploy an Amazon EKS cluster, and we dynamically scale GPU nodes using Karpenter when the model is deployed using inference charts.

### Prerequisites

Before we begin, ensure you have all the necessary prerequisites in place to make the deployment process smooth. Make sure you have installed the following tools on your machine:

1. [aws cli](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html)
2. [kubectl](https://Kubernetes.io/docs/tasks/tools/)
3. [terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)
4. [envsubst](https://pypi.org/project/envsubst/)

### Deploy

Clone the repository

```bash
git clone https://github.com/awslabs/ai-on-eks.git
```

**Important Note:**

**Step1**: Ensure that you update the region in the `blueprint.tfvars` file before deploying the blueprint.
Additionally, confirm that your local region setting matches the specified region to prevent any discrepancies.

For example, set your `export AWS_DEFAULT_REGION="<REGION>"` to the desired region:


**Step2**: Run the installation script.

```bash
cd ai-on-eks/infra/envoy-ai-gateway/ && chmod +x install.sh
./install.sh
```
### Verify the resources

Once the installation finishes, verify the Amazon EKS Cluster.

Creates k8s config file to authenticate with EKS.

```bash
aws eks --region us-west-2 update-kubeconfig --name envoy-gateway-cluster
```

```bash
kubectl get nodes
```

```text
NAME                                           STATUS   ROLES    AGE      VERSION
ip-100-64-118-130.us-west-2.compute.internal   Ready    <none>   3h9m     v1.33.5-eks-ba24e9c
ip-100-64-127-174.us-west-2.compute.internal   Ready    <none>   3h9m     v1.33.5-eks-ba24e9c
ip-100-64-132-168.us-west-2.compute.internal   Ready    <none>   3h9m     v1.33.5-eks-ba24e9c
```

Verify the Karpenter autoscaler Nodepools

```bash
kubectl get nodepools
```

```text
NAME                NODECLASS           NODES   READY   AGE
g5-gpu-karpenter    g5-gpu-karpenter    0       True    5d20h
g6-gpu-karpenter    g6-gpu-karpenter    0       True    5d20h
g6e-gpu-karpenter   g6e-gpu-karpenter   2       True    5d20h
inferentia-inf2     inferentia-inf2     0       True    5d20h
trainium-trn1       trainium-trn1       0       True    5d20h
x86-cpu-karpenter   x86-cpu-karpenter   1       True    5d20h
```

```bash
kubectl get pod,svc,deployment -n envoy-gateway-system
```

```text
NAME                                                          READY   STATUS    RESTARTS   AGE
pod/envoy-default-ai-gateway-27dc8f39-7595568bc8-8cxsx        3/3     Running   0          24h
pod/envoy-gateway-587c57f58f-hql9n                            1/1     Running   0          24h

NAME                                             TYPE           CLUSTER-IP       EXTERNAL-IP                                                                     PORT(S)                                            AGE
service/envoy-gateway                            ClusterIP      172.20.166.211   <none>                                                                          18000/TCP,18001/TCP,18002/TCP,19001/TCP,9443/TCP   5d20h

NAME                                                     READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/envoy-default-ai-gateway-27dc8f39        1/1     1            1           5d20h
deployment.apps/envoy-gateway                            1/1     1            1           5d20h
```

</CollapsibleContent>

## Deploying self-hosted models

Envoy AI gateway currently supports self-hosted models that can speak OpenAI API schema. We will deploy two models using [AI on EKS Inference Charts](https://github.com/awslabs/ai-on-eks-charts).

### 1. Create Hugging Face Token Secret

Create a Kubernetes secret with your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens):

```bash
kubectl create secret generic hf-token --from-literal=token=your_huggingface_token
```

### 2. Deploy Pre-configured Models

Choose from the available pre-configured models and deploy:

:::warning

These deployments will need GPU/Neuron resources which need to
be [enabled](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-resource-limits.html) and cost more than CPU only
instances.

:::

```bash
# Add helm chart repository
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update
```

```bash
# Model 1: Deploy qwen3 model
helm install qwen3-1.7b ai-on-eks/inference-charts -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-qwen3-1.7b-vllm.yaml \
  --set nameOverride=qwen3 \
  --set fullnameOverride=qwen3 \
  --set inference.serviceName=qwen3

# Model 2: Deploy gpt oss model
helm install gpt-oss ai-on-eks/inference-charts -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-gpt-oss-20b-vllm.yaml \
  --set nameOverride=gpt-oss \
  --set fullnameOverride=gpt-oss \
  --set inference.serviceName=gpt-oss
```

Verify models up and running

```bash
kubectl get pod
```

```text
NAME                       READY   STATUS    RESTARTS   AGE
gpt-oss-67c5bcdd5c-lx9vh   1/1     Running   0          44h
qwen3-b5fdf6bd5-cxkwd      1/1     Running   0          44h
```

### Enabling access to AWS Bedrock models

Access to all Amazon Bedrock foundation models is enabled by default with the correct AWS Marketplace permissions. For more details on managing access to models in Amazon Bedrock, review [Access Amazon Bedrock foundation models](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)

Review [Supported foundation models in Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for a list of foundation models available via Amazon Bedrock and relevant model information including Model ID.

## Multi-model routing

Route requests to different AI models based on the `x-ai-eg-model` header. This header enables Envoy AI gateway to identify appropriate route configured within the gateway and routes client traffic to relevant backend kubernetes service. In this case, it's a service that exposes a self-hosted model or Amazon Bedrock model.

### Deploy common gateway infrastructure

```bash
cd ../../blueprints/gateways/envoy-ai-gateway
kubectl apply -f gateway.yaml
```

```text
serviceaccount/ai-gateway-dataplane-aws created
gatewayclass.gateway.networking.k8s.io/envoy-gateway created
envoyproxy.gateway.envoyproxy.io/ai-gateway created
gateway.gateway.networking.k8s.io/ai-gateway created
clienttrafficpolicy.gateway.envoyproxy.io/ai-gateway-buffer-limit created
```

### Configure model backends

```bash
kubectl apply -f model-backends.yaml
```

```text
backend.gateway.envoyproxy.io/gpt-oss-backend created
aiservicebackend.aigateway.envoyproxy.io/gpt-oss created
backend.gateway.envoyproxy.io/qwen3-backend created
aiservicebackend.aigateway.envoyproxy.io/qwen3 created
backend.gateway.envoyproxy.io/bedrock-backend created
aiservicebackend.aigateway.envoyproxy.io/bedrock created
backendsecuritypolicy.aigateway.envoyproxy.io/bedrock-policy created
backendtlspolicy.gateway.networking.k8s.io/bedrock-tls created
```

### Configure model routes

```bash
kubectl apply -f multi-model-routing/ai-gateway-route.yaml
```

```text
aigatewayroute.aigateway.envoyproxy.io/multi-model-route created
```

## Test

```bash
python3 multi-model-routing/client.py
```

**Expected Output**:
```
🚀 AI Gateway Multi-Model Routing Test
============================================================
Gateway URL: http://k8s-envoygat-envoydef-xxxxxxxxxx-xxxxxxxxxxxxxxxx.elb.us-west-2.amazonaws.com

=== Testing Qwen3 1.7B ===
Status Code: 200
✅ SUCCESS: Qwen3 - [response content]

=== Testing Self-hosted GPT ===
Status Code: 200
✅ SUCCESS: GPT - [response content]

=== Testing Bedrock Claude ===
Status Code: 200
✅ SUCCESS: Bedrock Claude - [response content]

🎯 Final Results:
• Qwen3 1.7B: ✅ PASS
• GPT OSS 20B: ✅ PASS
• Bedrock Claude: ✅ PASS

📊 Summary: 3/3 models working
```

## Rate limiting

Token-based rate limiting with automatic tracking for AI workloads.

**Features**:
- Token-based rate limiting (input, output, and total tokens)
- User-based rate limiting using `x-user-id` header
- Redis backend for distributed rate limiting (automatically deployed)
- Configurable limits per user per time window

### Configure rate limiting

```bash
kubectl apply -f rate-limiting/ai-gateway-route.yaml
kubectl apply -f rate-limiting/ai-gateway-rate-limit.yaml
kubectl apply -f rate-limiting/backend-traffic-policy.yaml
```

### Test rate limiting

```bash
python3 rate-limiting/client.py
```

## Configuration Details

### Routing Configuration
The AI Gateway routes requests based on the `x-ai-eg-model` header:

| Header Value | Backend | Endpoint | Model Type |
|--------------|---------|----------|------------|
| `Qwen/Qwen3-1.7B` | qwen3 | `/v1/chat/completions` | Self-hosted |
| `openai/gpt-oss-20b` | gpt-oss | `/v1/chat/completions` | Self-hosted |
| `anthropic.claude-3-haiku-20240307-v1:0` | bedrock | `/anthropic/v1/messages` | AWS Bedrock |

### Bedrock Integration Details
- **Authentication**: Pod Identity (automatically configured via installation script)
- **Schema**: AWSAnthropic for native Bedrock support
- **Endpoint**: `/anthropic/v1/messages` (Anthropic Messages API format)
- **Region**: Configurable in `backend-security-policy.yaml` (default: us-west-2)

## Resources

- [Envoy AI Gateway Documentation](https://github.com/envoyproxy/ai-gateway)
- [Envoy Gateway Documentation](https://gateway.envoyproxy.io/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

## Important Notes

- **Multi-Model Routing**: Requires deployed AI model services and AWS Bedrock access
- **Rate Limiting**: Requires actual AI models that return real token usage data and Redis for storage
- **Bedrock Integration**: Requires AWS Bedrock API access, proper IAM setup, and Pod Identity configuration
- **Authentication**: Pod Identity for Bedrock is automatically configured when deploying via the installation script

These are working configuration examples that demonstrate AI Gateway capabilities with real AI model deployments and AWS Bedrock integration.
