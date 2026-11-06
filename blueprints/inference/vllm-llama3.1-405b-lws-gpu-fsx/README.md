# Multi-node inference over FSx for Lustre + EFA — Llama 3.1 405B (vLLM-OpenAI)

Very large language models — hundreds of billions of parameters — don't fit on a single GPU node, and
their weights (hundreds of GB) take minutes to load from storage at pod start, leaving expensive GPUs
idle on every cold start and scale-up. This blueprint serves such a model across multiple GPU nodes on
Amazon EKS and loads its weights quickly from **Amazon FSx for Lustre over EFA (RDMA)** — for teams
running large-model inference on EKS who want to cut cold-start time and GPU-idle cost. The worked
example is Llama-3.1-405B; the pattern applies to any large multi-node model.

Serve Llama-3.1-405B across **2× p5.48xlarge (16 GPUs, TP8 × PP2)** on EKS, with model
weights streamed from **Amazon FSx for Lustre over EFA (RDMA)** using the Run:AI Model
Streamer, orchestrated by **LeaderWorkerSet + Ray**.

EFA removes the single-flow TCP ceiling on the weight read, which is the dominant cost of
cold-starting a very large model. A model this size does not fit on one 8-GPU node, so it is
split across two nodes (tensor-parallel within a node over NVLink, pipeline-parallel across
nodes).

This blueprint ships two image variants:

| Variant | Base image | Inter-node NCCL | Use when |
|---|---|---|---|
| **vLLM-OpenAI** (this dir) | `vllm/vllm-openai` | TCP (no aws-ofi-nccl) | Default; simplest public image |
| **DLC** (`../vllm-llama3.1-405b-lws-gpu-fsx-dlc/`) | AWS Deep Learning Container | **EFA** (aws-ofi-nccl) | You want NCCL-over-EFA too |

> **FSx weight-loading rides EFA in both variants** — it is a node/LNet property, independent of
> the image. The variants differ only in the *inter-node GPU* transport (NCCL).

## Prerequisites

1. **Base deployed with the FSx-EFA options on.** Use the infra wrapper
   (`infra/vllm-llama3.1-405b-lws-gpu-fsx/terraform/blueprint.tfvars`), which sets:
   `deploy_fsx_volume=true`, `fsx_efa_enabled=true`, `enable_aws_fsx_csi_driver=true`, and an
   EFA GPU node pool. The base creates the EFA-enabled FSx filesystem and the static
   `fsx-static-pvc` this workload mounts.
2. **Model weights staged on FSx** at `/mnt/fsx/models/Llama-3.1-405B-Instruct` (one-time copy
   via a Job that mounts the FSx PVC on a non-GPU node).
3. **A custom image** built and pushed to your ECR (next section).
4. **LeaderWorkerSet controller** installed (the base does not ship it) — only required for the
   `kubectl` deploy path; the Terraform path installs it for you.

   ```bash
   LWS_VERSION=v0.9.0   # validated pin; newer up to v1.0.0 exist — see https://github.com/kubernetes-sigs/lws/releases
   kubectl apply --server-side -f \
     https://github.com/kubernetes-sigs/lws/releases/download/${LWS_VERSION}/manifests.yaml
   ```

## Build the image

A thin 2-line layer on top of a vLLM base — Ray (multi-node coordination) + the Run:AI Model
Streamer (fast FSx weight load, `--load-format runai_streamer`). You do **not** rebuild vLLM.

```bash
cd image
export ACCOUNT_ID=<your-12-digit-account>
export REGION=<your-region>
export ECR_REPO=<your-repo>            # e.g. vllm-405b
export IMAGE_TAG=<your-tag>            # e.g. v0.26.0-runai
export BASE_IMAGE=<your-vllm-image>    # default vllm/vllm-openai:v0.26.0 (validated pin; newer tags on Docker Hub)
./build-kaniko.sh                      # in-cluster Kaniko build — no local Docker needed
```

Prereqs: kubectl context on the cluster, AWS creds, an ECR repo you can push to, and a CPU
(non-GPU) node for the build. The script prints the resulting image URI.

**Adapting to your own vLLM image:** point `BASE_IMAGE` at your existing vLLM image. Use
`--no-deps` on the streamer so it can't change your torch/vLLM, and pin Ray to a version
compatible with your vLLM release. The `vllm/vllm-openai` base ships `python3` (no `python`
symlink) and no Ray — verify with
`python3 -c "import vllm, torch, ray; print(vllm.__version__)"`.

## Deploy

### Path A — kubectl

1. Install the LWS controller (see Prerequisites #4).
2. Replace `<YOUR_ECR_IMAGE>` in `vllm-405b-lws-gpu-fsx.yaml` (both leader and worker) with the
   image URI from the build step.
3. Apply:

   ```bash
   kubectl apply -f vllm-405b-lws-gpu-fsx.yaml
   ```

### Path B — Terraform

Installs the LWS controller, renders the manifest from your variables, and applies it.

```bash
cd terraform
terraform init
terraform apply \
  -var region=<your-region> \
  -var cluster_name=<your-cluster> \
  -var vllm_image=<ACCOUNT>.dkr.ecr.<region>.amazonaws.com/vllm-405b:v0.26.0-runai
```

Key variables (see `terraform/main.tf` for all): `vllm_image` (required), `gpu_instance_type`
(`p5.48xlarge`), `worker_replicas` (1 → `size`=2 nodes), `tensor_parallel_size` (8),
`pipeline_parallel_size` (2), `model_path`, `pvc_name` (`fsx-static-pvc`).

## Test

```bash
kubectl get pods -l app=vllm-405b            # leader + worker Running, leader Ready after weight load
kubectl exec <leader-pod> -- \
  curl -s http://llm405b-leader.default.svc.cluster.local:8000/v1/models
```

First readiness takes several minutes (weight load + CUDA-graph capture) — the readiness probe
allows for this.

## Design notes & caveats

- **Namespace:** the pods mount `fsx-static-pvc`, which lives in the base's `fsx_pvc_namespace`
  (default `default`). PVCs are namespaced — if you change that base variable, change this
  workload's `namespace` to match, or the pods can't bind the claim.
- **`privileged: true`** is required for EFA/RDMA + GPUDirect device access. If your cluster
  policy forbids privileged pods, you will need an exception for this workload.
- **Hardcoded GPU count (kubectl path):** the plain manifest's Ray wait-loop expects `16` GPUs
  (2 nodes × 8). If you change `size`, update that number too — or use the Terraform path, which
  computes it from your variables.
- **`max-model-len` defaults to 8192** for a predictable serving footprint; raise it for longer
  contexts if you have the KV-cache headroom.
- **Gotchas baked into the manifest:** node-local `VLLM_CACHE_ROOT` (vLLM hangs on a Lustre-backed
  cache), `do-not-disrupt` (don't evict a running group), no hugepages (NCCL-on-TCP path), and a
  Service named off the `vllm-*` prefix (avoids a Kubernetes env-var collision). Each is commented
  inline in `vllm-405b-lws-gpu-fsx.yaml`.
