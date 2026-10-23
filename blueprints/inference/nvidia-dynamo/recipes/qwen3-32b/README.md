# Qwen3-32B Recipe

Model: [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B)

## Sync model weights

Follow the [model-cache](../../model-cache/README.md) guide to sync the Qwen3-32B weights from HuggingFace to either a `model-cache` PVC or an Amazon S3 bucket. Both workflows are documented there.

When running the [HuggingFace-to-S3](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) sync job, use the following values in the `s3ModelCopy` configuration. These are tuned for the size of Qwen3-32B:

| Parameter | Recommended value | Why |
| --- | --- | --- |
| `storageSize` | `100` | Scratch volume in GiB for the download. 100 GiB comfortably holds the full Qwen3-32B weights. |
| `requireNetworkBandwidth` | `25000` | Schedules the Job onto a node with at least 25 Gbps of network bandwidth, so the HuggingFace-to-S3 transfer isn't network-bound. |
| `parameters.fileWorkers` | `18` | Downloads up to 18 files in parallel to maximize throughput. There are 18 safetensors files with a max size of ~4GB each, so `storageSize` must scale accordingly - see the [sizing note](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) in the model-cache guide. |

```bash
export MODEL_NAME="Qwen/Qwen3-32B"
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

> Total time: 54.4s

## Deploy Model

```bash
envsubst '${S3_BUCKET} ${INFERENCE_SA}' < vllm/h100-agg-mx-s3.yaml | kubectl apply -n dynamo-system -f -

# Watch logs

kubectl -n dynamo-system logs -l  nvidia.com/dynamo-graph-deployment-name=qwen3-32b-h100-agg-mx-s3-vllm --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f

# Delete
envsubst '${S3_BUCKET} ${INFERENCE_SA}' < vllm/h100-agg-mx-s3.yaml | kubectl delete -n dynamo-system -f -
```
