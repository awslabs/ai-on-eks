# nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 Recipe

Model: [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16)

## Sync model weights

Follow the [model-cache](../../model-cache/README.md) guide to sync the nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 weights from HuggingFace to either a `model-cache` PVC or an Amazon S3 bucket. Both workflows are documented there.

For this model, the recommended path is [Amazon S3 bucket](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3).

When running the [HuggingFace-to-S3](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) sync job, use the following values in the `s3ModelCopy` configuration. These are tuned for the size of nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:

| Parameter | Recommended value | Why |
| --- | --- | --- |
| `storageSize` | `200` | Scratch volume in GiB for the download. |
| `requireNetworkBandwidth` | `25000` | Schedules the Job onto a node with at least 25 Gbps of network bandwidth, so the HuggingFace-to-S3 transfer isn't network-bound. |
| `parameters.fileWorkers` | `35` | Downloads up to 35 files in parallel to maximize throughput. There are 225 safetensors files with a max size of ~5GB each, so `storageSize` must scale accordingly - see the [sizing note](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) in the model-cache guide. |

```bash
export MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
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
  storageSize: 200
  requireNetworkBandwidth: 25000
  parameters:
    fileWorkers: 35
EOF
```

> Total time: 943.5s

## Deploy Model

```bash
envsubst '${S3_BUCKET} ${INFERENCE_SA}' < vllm/b300-disagg-mx-s3.yaml | kubectl apply -n dynamo-system -f -

# Watch all logs
kubectl -n dynamo-system logs -l nvidia.com/dynamo-graph-deployment-name=nemotron-3-ultra-b300-disagg-vllm --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f

# Delete
envsubst '${S3_BUCKET} ${INFERENCE_SA}' < vllm/b300-disagg-mx-s3.yaml | kubectl delete -n dynamo-system -f -
```
