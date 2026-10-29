# NVIDIA Dynamo Platform

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) is an inference open-source stack for serving large language models above inference engines. \
This directory contains recipes for deploying different inference backends (for example vLLM) and models with NVIDIA Dynamo on Amazon EKS.

The workflow has three phases:

1. **Deploy infrastructure** — provision the Amazon EKS cluster and supporting platform components with Terraform.
2. **Prepare the environment** — configure `kubectl`, wait for the platform to converge, and create the ServiceAccounts, secrets, and storage that recipes depend on.
3. **Run a recipe** — sync a model's weights and deploy it. Each model lives under [`recipes/`](./recipes/) (for example [`recipes/qwen3-32b`](./recipes/qwen3-32b/README.md)), and model-weight caching is covered in the [model-cache guide](./model-cache/README.md).

All commands below assume you are running from this directory (`blueprints/inference/nvidia-dynamo`), since the relative paths to the Terraform stack (`../../../infra/nvidia-dynamo`) are resolved from here.

## Capacity Requirements

We recommend using [EC2 Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-capacity-blocks.html) to guarantee availability for your deployment window. When purchasing a Capacity Block, **tag the reservation with `ai-on-eks=true`** — Karpenter's GPU NodePool is configured to discover and launch instances into Capacity Blocks matching this tag, ensuring your reserved capacity is automatically utilized when pods are scheduled.

## Deploy Infrastructure

Provision the EKS cluster and the Dynamo platform. The `install.sh` script copies the shared Terraform base into a local `_LOCAL` workspace, bootstraps the backend, and runs `terraform apply`. \
This creates the cluster, the `dynamo-system` namespace, the S3 model bucket, IAM roles for model sync and inference, EFS storage, and the ArgoCD applications that install the platform and dependencies add-ons.

Switch into the Terraform stack:

```bash
pushd ../../../infra/nvidia-dynamo
```

Edit `terraform/blueprint.tfvars` to match your environment. The most common changes are the AWS region and the EKS control plane version.

You can control the deployment region in any of the following ways:
* Edit the `blueprint.tfvars` file and modify the `region` variable.
* Set the `AWS_REGION=<AWS_REGION>` environment variable.

> Default AWS region is `us-west-2`

Run the installer:

```bash
./install.sh
```

Once the apply completes, return to the blueprint directory:

```bash
popd
```

This step can take 20-30 minutes on a first run while the cluster and node groups come up.

## Prepare environment

### 1. Configure kubectl

Point your local `kubectl` at the newly created cluster. \
The Terraform stack exposes a ready-made `aws eks update-kubeconfig` command as the `configure_kubectl` output, so this writes the cluster context into your kubeconfig:

```bash
eval $(terraform -chdir="../../../infra/nvidia-dynamo/terraform/_LOCAL" output -raw configure_kubectl)
```

### 2. Wait for ArgoCD applications to be healthy

The platform add-ons (CSI drivers, GPU operator, Dynamo operator, KAI scheduler, Grove, and so on) are delivered as ArgoCD applications. Wait for every application to report `Healthy` before continuing, otherwise later steps may fail because a dependency (such as the EFS CSI driver or the Dynamo CRDs) is not yet ready. This waits up to 10 minutes:

```bash
kubectl wait --for=jsonpath='{.status.health.status}'=Healthy \
  applications.argoproj.io --all \
  -n argocd \
  --timeout=600s
```

### 3. Set environment variables

Read the S3 bucket name and the two ServiceAccount names back out of the Terraform outputs and export them into your shell. \
The recipes and the model-cache guide reference these variables (`$S3_BUCKET`, `$MODEL_SYNC_SA`, `$INFERENCE_SA`) directly, so export them in the same shell you will run those steps from:

```bash
eval "$(
  terraform -chdir="../../../infra/nvidia-dynamo/terraform/_LOCAL" output -json | jq -r '
    "export S3_BUCKET=" + (.s3_models_buckets_name.value[0]),
    "export MODEL_SYNC_SA=" + (.s3_models_sync_sa.value),
    "export INFERENCE_SA=" + (.s3_models_inference_sa.value)
  '
)"
```

- `S3_BUCKET` — the bucket where model weights are stored.
- `MODEL_SYNC_SA` — the ServiceAccount used by the sync Job to **write** weights to S3.
- `INFERENCE_SA` — the ServiceAccount used by inference Pods to **read** weights from S3.

### 4. Create Service Accounts for syncing and using models from S3

Create the two ServiceAccounts in the `dynamo-system` namespace. \
Terraform maps each of these names to an IAM role (via IRSA or EKS Pod Identity): the sync SA gets write access to the bucket and the inference SA gets read access. \
Splitting them keeps least-privilege boundaries between the sync workload and the serving workload.

```bash
kubectl -n dynamo-system create sa $MODEL_SYNC_SA
kubectl -n dynamo-system create sa $INFERENCE_SA
```

### 5. Create HuggingFace secret with HuggingFace token

Model downloads pull from HuggingFace, which requires a token for gated models and for higher rate limits. Store your token in a Secret named `hf-token-secret`; the download and sync Jobs read it to authenticate. Replace `<HF TOKEN>` with your own token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

```bash
export HF_TOKEN=<HF TOKEN>
kubectl -n dynamo-system create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=${HF_TOKEN}
```

### 6. Create EFS Persistent Volume Claims (PVCs)

Create the shared EFS-backed PVCs that Dynamo uses as caches. All three use the `efs-sc-dynamic` storage class with `ReadWriteMany` access so they can be shared across Pods:

- `model-cache` — holds model weights when using the PVC-based caching workflow.
- `compilation-cache` — caches compiled kernels/engines so replicas don't recompile on every start.
- `perf-cache` — caches performance-tuning or benchmarks artifacts.

```bash
kubectl -n dynamo-system apply -f pvc-efs.yaml
```

## NVIDIA ModelExpress

[ModelExpress](https://github.com/ai-dynamo/modelexpress) accelerates model distribution by tracking model metadata and cache entries in-cluster and enabling peer-to-peer weight transfer between Pods. It is optional — use it when you want faster, coordinated model loading across replicas. Enabling it requires installing its CRDs and RBAC.

### ModelExpress CRDs

Install the custom resource definitions that ModelExpress uses to track models. This creates the `modelmetadatas.modelexpress.nvidia.com` and `modelcacheentries.modelexpress.nvidia.com` CRDs:

```bash
kubectl apply -f https://raw.githubusercontent.com/ai-dynamo/modelexpress/refs/heads/main/examples/crds.yaml
```

### ModelExpress `serviceAccount` and `RBAC`

Grant ModelExpress the permissions it needs to read and write those custom resources. This creates the `modelexpress` Kubernetes `serviceAccount`, `Role`, `RoleBinding`, and `ClusterRole`:

```bash
kubectl -n dynamo-system apply -f \
 https://raw.githubusercontent.com/ai-dynamo/modelexpress/refs/heads/main/examples/dynamo_p2p_transfer_k8s/vllm/rbac-modelmetadata.yaml
```

## Deploy a model

With the environment prepared, pick a recipe under [`recipes/`](./recipes/) and follow its README. Each recipe documents the model-specific sync values and the deployment manifest. See the [model-cache guide](./model-cache/README.md) for the shared weight-caching workflows (PVC or S3) that the recipes build on.

## Cleanup

Tear down the cluster and all provisioned infrastructure. This destroys the Terraform-managed resources created by `install.sh`, so make sure any model weights or data you want to keep are backed up first (the S3 bucket and its contents are removed as part of the destroy).

```bash
pushd ../../../infra/nvidia-dynamo
./cleanup.sh
popd
```
