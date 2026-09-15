# Deploying Kimi K3 on Amazon EKS

Deploy [Moonshot AI's Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) on Amazon EKS using vLLM.

## Prerequisites

- An [Inference-Ready EKS Cluster](../../) deployed and running
- `kubectl` configured to access the cluster
- A [Hugging Face token](https://huggingface.co/settings/tokens) with access to the model
- `helm` CLI installed (v3+)

## Capacity Requirements

This recipe of Kimi K3 requires **1 x p6-b300 instance** (8 x B300 GPUs) for inference.
We recommend using [EC2 Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) to guarantee availability for your deployment window. When purchasing a Capacity Block, **tag the reservation with `ai-on-eks=true`** — Karpenter's GPU NodePool is configured to discover and launch instances into Capacity Blocks matching this tag, ensuring your reserved capacity is automatically utilized when pods are scheduled.

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/awslabs/ai-on-eks.git
cd ai-on-eks/infra/solutions/inference-ready-cluster
```

## Step 2: Deploy the Infrastructure

Review and adjust `terraform/blueprint.tfvars` for your environment, then deploy:

```bash
./install.sh
```

Once complete, configure `kubectl`:

```bash
eval $(terraform -chdir="./terraform/_LOCAL" output -raw configure_kubectl)
```

Wait for all ArgoCD applications to become healthy:

```bash
kubectl wait --for=jsonpath='{.status.health.status}'=Healthy \
  applications.argoproj.io --all \
  -n argocd \
  --timeout=600s
```

## Step 3: Prepare the Environment

Export the Terraform outputs needed for subsequent steps:

```bash
eval "$(
  terraform -chdir="./terraform/_LOCAL" output -json | jq -r '
    "export S3_BUCKET=" + (.s3_models_buckets_name.value[0]),
    "export MODEL_SYNC_SA=" + (.s3_models_sync_sa.value),
    "export INFERENCE_SA=" + (.s3_models_inference_sa.value)
  '
)"
```

Create service accounts with for S3 access:

```bash
kubectl -n default create sa $MODEL_SYNC_SA
kubectl -n default create sa $INFERENCE_SA
```

## Step 4: Create the Hugging Face Secret

```bash
kubectl -n default create secret generic hf-token \
  --from-literal=token=<YOUR_HF_TOKEN>
```

> Replace `<YOUR_HF_TOKEN>` with your actual [Hugging Face token](https://huggingface.co/settings/tokens).

## Step 5: Download Model Weights to S3

Add the AI on EKS Helm chart repository:

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update
```

Launch the S3 copy job. This downloads Kimi K3 weights from Hugging Face to your S3 bucket using NVMe-backed storage for throughput:

```bash
export REPO_ID="moonshotai/Kimi-K3"

helm install s3-copy-moonshot-kimi-k3 ai-on-eks/inference-charts \
  -n default -f - <<EOF
s3ModelCopy:
  namespace: default
  model: $REPO_ID
  s3Bucket: $S3_BUCKET
  serviceAccountName: $MODEL_SYNC_SA
  requireLocalNvme: true
  storageSize: 600
  requireNetworkBandwidth: 25000
  parameters:
    fileWorkers: 35
EOF
```

> This job runs on an NVMe-equipped instance and uses 35 parallel workers to maximize download speed. Storage is set to 600 GiB to accommodate the full model weights.

Monitor progress:

```bash
kubectl logs -n default -f -l app.kubernetes.io/name=s3-model-copy,app.kubernetes.io/instance=s3-copy-moonshot-kimi-k3
```

Once the job completes successfully, clean up the copy resources:

```bash
helm uninstall s3-copy-moonshot-kimi-k3 -n default
```

## Step 6: Deploy vLLM for Inference

Deploy vLLM with the RunAI Model Streamer to load weights directly from S3 in a distributed fashion:

```bash
helm upgrade --install kimi-k3-vllm ai-on-eks/inference-charts -n default \
  -f https://raw.githubusercontent.com/awslabs/ai-on-eks-charts/refs/heads/main/charts/inference-charts/values-kimi-k3-vllm-b300.yaml \
  --set modelPath="s3://${S3_BUCKET}/${REPO_ID}" \
  --set serviceAccountName="${INFERENCE_SA}" \
  --set modelParameters.loadFormat="runai_streamer" \
  --set modelParameters.modelLoaderExtraConfig="'{\"distributed\":true}'"
```

Verify the deployment is running:

```bash
kubectl -n default get pods -l ai.eks.amazonaws.com/name=kimi-k3-vllm --watch
```

> The deployment can take up to 15 minutes to become ready. This includes node provisioning, streaming model weights from S3, graph compilation, and CUDA kernel warm-up.

Monitor progress by tailing the logs:

```bash
kubectl -n default logs -f -l ai.eks.amazonaws.com/name=kimi-k3-vllm
```

## Validation

Once the pods are ready, test the endpoint:

```bash
kubectl -n default port-forward svc/kimi-k3-vllm 8000:8000
```

In a separate terminal:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K3",
    "messages": [{"role": "user", "content": "What is Mixture of Experts?"}],
    "max_tokens": 256
  }'
```

## Cleanup

Remove the inference deployment:

```bash
helm uninstall kimi-k3-vllm -n default
```

To remove all infrastructure:

```bash
./cleanup.sh
```
