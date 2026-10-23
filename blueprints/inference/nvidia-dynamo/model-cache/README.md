# Model Caching

Model caching pre-fetches model weights from HuggingFace so that inference Pods start quickly without each replica downloading the weights independently from HuggingFace. This page covers two caching targets:

- **PVC** — download the weights once into a shared `model-cache` PersistentVolumeClaim that inference Pods mount directly. Simplest to consume, since the weights appear as a shared filesystem path with no S3 client or IAM setup required. The tradeoff is throughput: the PVC is typically backed by a shared storage EFS/FSxLustre, which is easy to share ReadWriteMany across Pods but might be slower and less cost effective than S3-to-NVMe or S3 direct streaming, so model load times can be higher.
- **Amazon S3** — sync the weights to an S3 bucket that inference Pods access through a Kubernetes ServiceAccount mapped to an IAM role (via IRSA or EKS Pod Identity) granting read access to the bucket. Best for durable storage that can be shared across clusters, regions, or accounts, most cost effective.

## Prerequisites

- A running EKS cluster with the `dynamo-system` namespace and the Dynamo blueprint installed.
- A HuggingFace access token stored in a Kubernetes Secret named `hf-token-secret` in the `dynamo-system` namespace. The download job reads this token to authenticate with HuggingFace (required for gated models and higher rate limits). Create it with:

  ```bash
  kubectl -n dynamo-system create secret generic hf-token-secret \
    --from-literal=HF_TOKEN=<YOUR_HF_TOKEN>
  ```

- The `model-cache` PVC must already exist in the `dynamo-system` namespace (for the PVC workflow). Confirm with `kubectl -n dynamo-system get pvc model-cache`.

## Sync model weights from HuggingFace to PVC

This workflow runs a Kubernetes Job ([`model-download.yaml`](./model-download.yaml)) that pulls a model repository from HuggingFace and writes it into the shared `model-cache` PVC.
It enables `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_XET_HIGH_PERFORMANCE=1` for faster, parallelized downloads.

### 1. Create a model download job

Set `MODEL_NAME` to the HuggingFace repository ID (for example, `Qwen/Qwen3-32B`), then render and apply the manifest. `envsubst` substitutes `${MODEL_NAME}` into the manifest before it is applied:

```bash
export MODEL_NAME=<MODEL NAME>
envsubst '${MODEL_NAME}' < model-download.yaml | kubectl apply -n dynamo-system -f -
```

### 2. Monitor

Wait for the Job to allocate a Pod. The Pod requests 2 CPU / 64 GiB (up to 8 CPU), so the scheduler may need to provision a node first; the command below waits up to 10 minutes:

```bash
kubectl -n dynamo-system wait --for=condition=ready pod --selector=job-name=model-download --timeout=600s
```

Watch the Job logs to follow download progress. You should see `pip install` output followed by `hf download` transferring the model files into `/models`:

```bash
kubectl -n dynamo-system logs -l app=model-download -f
```

Confirm the Job completed successfully before cleaning up. Because `restartPolicy: Never` and `backoffLimit: 3` are set, a persistent failure surfaces as a `Failed` condition rather than an endless retry loop:

```bash
kubectl -n dynamo-system get job/model-download
```

### 3. Cleanup

Once the download Job has completed, delete it to free the Pod and clear the Job object. The weights remain in the `model-cache` PVC:

```bash
kubectl -n dynamo-system delete job/model-download
```

## Sync model weights from HuggingFace to Amazon S3

This workflow syncs a model repository from HuggingFace into an Amazon S3 bucket using the [AI on EKS inference-charts](https://github.com/awslabs/ai-on-eks-charts/tree/main/charts/inference-charts) Helm chart. The chart's `s3ModelCopy` component runs a one-shot Kubernetes Job that downloads the repository (including model weights) from HuggingFace and uploads it to your bucket. It schedules onto an NVMe-equipped, high-bandwidth instance and downloads files in parallel to maximize throughput. Inference Pods then pull the weights from S3 at startup (typically via streaming libraries directly from S3 or via initContainer by downloading from S3 to a local path) instead of re-downloading from HuggingFace, which makes the weights durable and reusable across clusters, regions, and accounts.

### Prerequisites

This workflow relies on the S3 bucket, ServiceAccount, and HuggingFace secret created during the blueprint setup (see the [nvidia-dynamo README](../README.md)). Confirm the following are in place before continuing:

- **Environment variables** — `S3_BUCKET` (destination bucket) and `MODEL_SYNC_SA` (the sync ServiceAccount) are exported in your shell. These are produced from the Terraform outputs:

  ```bash
  eval "$(
    terraform -chdir="../../../infra/nvidia-dynamo/terraform/_LOCAL" output -json | jq -r '
      "export S3_BUCKET=" + (.s3_models_buckets_name.value[0]),
      "export MODEL_SYNC_SA=" + (.s3_models_sync_sa.value)
    '
  )"
  ```

- **Sync ServiceAccount** — `$MODEL_SYNC_SA` exists in the `dynamo-system` namespace and is mapped to an IAM role (via IRSA or EKS Pod Identity) with write access to the bucket. The copy Job runs under this ServiceAccount to upload the weights.
- **HuggingFace token** — a Secret named `hf-token-secret` exists in `dynamo-system`. The Job reads it via `envFrom` to authenticate with HuggingFace (required for gated models and for higher rate limits).

### 1. Add the AI on EKS Helm chart repository

```bash
helm repo add ai-on-eks https://awslabs.github.io/ai-on-eks-charts/
helm repo update
```

### 2. Deploy S3 Copy job

Set `MODEL_NAME` to the HuggingFace repository ID (for example, `Qwen/Qwen3-32B`), then install the chart. The inline values below configure the `s3ModelCopy` Job with the destination bucket, the sync ServiceAccount, the HuggingFace token, and the node/throughput requirements:

```bash
export MODEL_NAME=<MODEL NAME>
helm -n dynamo-system install hf-s3-sync-model ai-on-eks/inference-charts -f - <<EOF
s3ModelCopy:
  namespace: dynamo-system
  model: $MODEL_NAME
  s3Bucket: $S3_BUCKET
  hfTokenSecret:
    name: "hf-token-secret"
    envFrom: true
  serviceAccountName: $MODEL_SYNC_SA
  requireLocalNvme: true
  storageSize: 100
  requireNetworkBandwidth: 25000
  parameters:
    fileWorkers: 18
EOF
```

Key values:

- `model` / `s3Bucket` — the HuggingFace repo to copy and the destination bucket. The weights land under `s3://$S3_BUCKET/$MODEL_NAME/`.
- `serviceAccountName` — the ServiceAccount (mapped to an IAM role) the Job assumes to write to S3.
- `hfTokenSecret` — the Secret supplying `HF_TOKEN`, injected via `envFrom` for HuggingFace authentication.
- `requireLocalNvme` / `requireNetworkBandwidth` — node constraints that place the Job on an NVMe-backed instance with at least 25 Gbps of network bandwidth for fast download and upload.
- `parameters.fileWorkers` — the number of files downloaded in parallel (18 here).

> This job runs on an NVMe-equipped instance and uses 18 parallel workers to maximize download speed. Storage is set to 100 GiB to accommodate the full model weights.
> `fileWorkers` drives the storage requirement: each worker downloads a file in parallel, so size the volume as `storageSize ≈ (largest file size in the repo) × fileWorkers`. Increasing `fileWorkers` speeds up the download but requires a larger `storageSize`.

### 3. Monitor

**1. Wait for the Job to allocate a Pod**

The Helm release creates a Kubernetes Job, which in turn schedules a Pod. Because the Pod requires an NVMe-backed instance with the requested network bandwidth, the scheduler may need to provision a new node first (e.g via Karpenter). This can take several minutes on the first run, which is why the command below waits up to 10 minutes for the Pod to become ready.

```bash
kubectl -n dynamo-system wait --for=condition=ready pod -l app.kubernetes.io/name=s3-model-copy,app.kubernetes.io/instance=hf-s3-sync-model --timeout=600s
```

If the command times out, inspect the Pod and its events to find out why it is not starting:

```bash
kubectl -n dynamo-system get pods -l app.kubernetes.io/name=s3-model-copy,app.kubernetes.io/instance=hf-s3-sync-model
kubectl -n dynamo-system describe pod -l app.kubernetes.io/name=s3-model-copy,app.kubernetes.io/instance=hf-s3-sync-model
```

The most common cause is that the Pod stays `Pending` because Karpenter cannot provision a node that satisfies the `requireLocalNvme: true` and `requireNetworkBandwidth: 25000` (25 Gbps) constraints. Check the Pod's `Events` section for a `FailedScheduling` message, and inspect the Karpenter controller logs for provisioning errors. Typical reasons Karpenter cannot launch a matching node:

- **Instance types not available in the region/AZ** — the NVMe-backed, high-bandwidth instance types that satisfy the constraints may not be offered in your region or the AZs your subnets span. Karpenter will report that it found no compatible instance types or has insufficient capacity.
- **Insufficient service quota** — your account may not have enough EC2 vCPU quota (for the relevant On-Demand or Spot instance family) to launch the node. Karpenter surfaces this as an insufficient-capacity or quota error; request a quota increase for the required instance family.
- **NodePool requirements exclude matching instances** — the Karpenter `NodePool`/`EC2NodeClass` may restrict instance families, sizes, capacity type (Spot vs On-Demand), or AZs such that no instance type meets both the NVMe and bandwidth requirements. Widen the NodePool requirements to include qualifying instance types.

If matching instance types genuinely aren't available to you, lower `requireNetworkBandwidth` first, then lower `storageSize` and `fileWorkers` accordingly (accepting slower throughput) so the Job can schedule on a broader set of instances.

**2. Watch the logs**

Follow the copy job's logs to track download and upload progress. You should see the model files being pulled from HuggingFace and streamed to your S3 bucket. Download time depends on model size, the number of `fileWorkers`, and the node's network bandwidth.

```bash
kubectl -n dynamo-system logs -f -l app.kubernetes.io/name=s3-model-copy,app.kubernetes.io/instance=hf-s3-sync-model
```

**3. Confirm the Job completed**

The `logs -f` command exits when the container finishes. Verify the Job reached the `Complete` condition rather than `Failed` before cleaning up:

```bash
kubectl -n dynamo-system get job -l app.kubernetes.io/instance=hf-s3-sync-model
```

Optionally, confirm the weights landed in S3:

```bash
aws s3 ls "s3://$S3_BUCKET/$MODEL_NAME/" --recursive --human-readable --summarize
```

### 4. Cleanup

Once the Job has completed successfully and the weights are present in S3, uninstall the Helm release to remove the copy resources (Job, Pod, and associated objects). This releases the NVMe node so it can scale down.

```bash
helm -n dynamo-system uninstall hf-s3-sync-model
```
