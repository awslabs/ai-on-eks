# Multi-node inference over FSx for Lustre + EFA — Llama 3.1 405B (AWS DLC)

Very large language models — hundreds of billions of parameters — don't fit on a single GPU node, and
their weights (hundreds of GB) take minutes to load from storage at pod start, leaving expensive GPUs
idle on every cold start and scale-up. This blueprint serves such a model across multiple GPU nodes on
Amazon EKS and loads its weights quickly from **Amazon FSx for Lustre over EFA (RDMA)** — for teams
running large-model inference on EKS who want to cut cold-start time and GPU-idle cost. The worked
example is Llama-3.1-405B; the pattern applies to any large multi-node model.

Serve Llama-3.1-405B across **2× p5.48xlarge (16 GPUs, TP8 × PP2)** on EKS, with weights streamed
from **Amazon FSx for Lustre over EFA (RDMA)**, orchestrated by **LeaderWorkerSet + Ray**. This is
the **AWS Deep Learning Container (DLC)** variant — it extends AWS's single-node vLLM DLC EKS example
to multi-node.

**What makes this variant different from the vLLM-OpenAI one
(`../vllm-llama3.1-405b-lws-gpu-fsx/`):** the DLC base ships **aws-ofi-nccl**, so **inter-node NCCL
also rides EFA** — not just the FSx weight load. Use this variant when you want NCCL-over-EFA for
the GPU-to-GPU traffic; use the vLLM-OpenAI variant for the simplest public image (NCCL-on-TCP).

| | this (DLC) | vLLM-OpenAI variant |
|---|---|---|
| Base image | AWS vLLM DLC | `vllm/vllm-openai` |
| Ray | added (thin layer) | added (thin layer) |
| aws-ofi-nccl (NCCL-over-EFA) | ✅ | ❌ (TCP) |
| FSx weight load over EFA | ✅ | ✅ |
| hugepages | requested (5120Mi) | not requested |

## Prerequisites

1. **Base deployed with the FSx-EFA options on** (infra wrapper `blueprint.tfvars`):
   `deploy_fsx_volume=true`, `fsx_efa_enabled=true`, `enable_aws_fsx_csi_driver=true`, and an EFA GPU
   node pool. The base creates the EFA-enabled FSx filesystem and the static `fsx-static-pvc`.
2. **Model weights staged on FSx** at `/mnt/fsx/models/Llama-3.1-405B-Instruct`.
3. **A custom DLC image** built + pushed to your ECR (next section).
4. **LeaderWorkerSet controller** installed (base does not ship it) — only for the `kubectl` path;
   the Terraform path installs it.
   ```bash
   LWS_VERSION=v0.9.0   # validated pin; newer up to v1.0.0 exist — see https://github.com/kubernetes-sigs/lws/releases
   kubectl apply --server-side -f \
     https://github.com/kubernetes-sigs/lws/releases/download/${LWS_VERSION}/manifests.yaml
   ```

## Build the image

The current AWS vLLM DLC (`public.ecr.aws/deep-learning-containers/vllm:server-cuda`) bundles
vLLM + EFA + aws-ofi-nccl (NCCL-over-EFA) **and** the Run:AI Model Streamer — but **not Ray**.
Multi-node vLLM needs Ray, so this adds a thin layer (Ray, plus a harmless no-op streamer install
as a safety net for older/custom bases) on top of the DLC base. You do **not** rebuild vLLM.

> This extends AWS's single-node
> [vLLM DLC EKS example](https://aws.github.io/deep-learning-containers/vllm/deployment/eks/) to
> multi-node (TP×PP via LeaderWorkerSet + Ray). Validated against the older 0.2x versioned DLC;
> re-validate against `server-cuda`. **Start by building the image on `server-cuda`** — the
> Dockerfile's final `import vllm, torch, ray` check fails the build if the pinned
> `ray[default]>=2.40,<2.50` range is incompatible with that DLC's vLLM; bump the ray pin if so,
> then proceed to the deploy test.

Pick a DLC base from the [AWS DLC vLLM gallery](https://aws.github.io/deep-learning-containers/vllm/):
- **Public ECR DLC (no login needed — easiest):**
  `public.ecr.aws/deep-learning-containers/vllm:server-cuda` — the current rolling tag. It is not
  version-pinned, so pin to a specific digest if you need reproducibility.
- **Private/canonical DLC:** see the gallery's **Image Access** page for the URI (requires
  `docker login` to the AWS DLC registry account).

### Option 1 — local Docker (x86 host)

```bash
export REGION=<your-region>
export ACCOUNT_ID=<your-12-digit-account>
export ECR_REPO=vllm-ray-runai
export IMAGE_TAG=dlc-ray-runai
export IMAGE_URI=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}
# DLC base — the current rolling tag (see the gallery: https://aws.github.io/deep-learning-containers/vllm/)
export DLC_BASE=public.ecr.aws/deep-learning-containers/vllm:server-cuda

# login to YOUR ECR to push (and to 763104351884 too, only if using the canonical DLC base)
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" 2>/dev/null || true

cd image
docker build --platform linux/amd64 --build-arg DLC_BASE="$DLC_BASE" -t "$IMAGE_URI" .
docker push "$IMAGE_URI"
```

### Option 2 — in-cluster Kaniko (no local Docker)

Use `image/build-kaniko.sh`. With the **public ECR DLC** base it needs no pull credential; with the
canonical `763104351884` base, add that account's pull credential to the Kaniko `config.json` secret.

## Deploy

### Path A — kubectl
1. Install the LWS controller (Prerequisites #4).
2. Replace `<YOUR_ECR_IMAGE>` in `vllm-405b-lws-gpu-fsx-dlc.yaml` (leader + worker) with your image URI.
3. `kubectl apply -f vllm-405b-lws-gpu-fsx-dlc.yaml`

### Path B — Terraform
```bash
cd terraform
terraform init
terraform apply \
  -var region=<your-region> \
  -var cluster_name=<your-cluster> \
  -var vllm_image=<ACCOUNT>.dkr.ecr.<region>.amazonaws.com/vllm-ray-runai:dlc-ray-runai
```

## Test

```bash
kubectl get pods -l app=vllm-405b
kubectl exec <leader-pod> -- \
  curl -s http://llm405b-leader.default.svc.cluster.local:8000/v1/models
```

First readiness takes several minutes (weight load + CUDA-graph capture).

## Design notes & caveats

- **NCCL-over-EFA** — confirm it engaged by checking the leader logs for an aws-ofi-nccl line
  (e.g. `NET/OFI ... efa-direct`) during init.
- **`VLLM_CACHE_ROOT` is on the FSx mount** (`/mnt/fsx/vllm_compile_cache/v1`), version-stamped — bump
  the `/v1` suffix on each image/vLLM upgrade so stale compiled kernels are not reused. (The
  vLLM-OpenAI variant uses node-local `/tmp` instead, because its newer vLLM hangs on a Lustre-backed
  cache — the DLC's vLLM does not.)
- **hugepages-2Mi (5120Mi)** is requested on this path; ensure the node/AMI exposes hugepages.
- **Namespace:** pods mount `fsx-static-pvc` in the base's `fsx_pvc_namespace` (default `default`);
  keep this workload's namespace in sync.
- **`privileged: true`** is required for EFA/RDMA + GPUDirect; needs a policy exception if your
  cluster forbids privileged pods.
- **Hardcoded GPU count (kubectl path):** the manifest's Ray wait-loop expects `16` (2×8). If you
  change `size`, update it — or use the Terraform path, which computes it.
- **Swap the base for your own approved EFA image** if required: override `DLC_BASE` with any base
  that carries EFA drivers + aws-ofi-nccl.
