# llm-d Cache-Aware Routing on EKS

This solution deploys precise KV-cache-aware inference routing on Amazon EKS using the Kubernetes Gateway API Inference Extension and llm-d, achieving up to 96% reduction in p90 TTFT compared to round-robin under sustained multi-turn load.

## Quick Start

```bash
export HF_TOKEN=<your-huggingface-token>
export KMS_KEY_ARN=<your-kms-key-arn>
./scripts/setup.sh
```

See the [full documentation](https://awslabs.github.io/ai-on-eks/docs/infra/inference/llm-d) on the AI on EKS website.

## What It Does

- Deploys 7x vLLM pods with prefix caching and ZeroMQ KVEvents publishing
- Installs Envoy AI Gateway + Envoy Gateway with InferencePool support
- Deploys llm-d precise prefix-cache scorer (EPP)
- Routes each request to the pod with the highest KV-cache affinity
- Includes a sustained multi-turn benchmark for validation

## Source Repository

The canonical source for this solution is: https://github.com/aws-samples/sample-eks-cache-aware-llm-routing

## Cost

~$10.49/hr (8x g5.2xlarge + EKS + networking). Scale to zero when not testing.

## Clean Up

```bash
./scripts/cleanup.sh
```
