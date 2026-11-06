# zai-org/GLM-5.3 Recipe

Model: [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)

## Sync model weights

Follow the [model-cache](../../model-cache/README.md) guide to sync the zai-org/GLM-5.3 weights from HuggingFace to either a `model-cache` PVC or an Amazon S3 bucket. Both workflows are documented there.

For this model, the recommended path is [Amazon S3 bucket](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3).

When running the [HuggingFace-to-S3](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) sync job, use the following values in the `s3ModelCopy` configuration. These are tuned for the size of zai-org/GLM-5.3:

| Parameter | Recommended value | Why |
| --- | --- | --- |
| `storageSize` | `200` | Scratch volume in GiB for the download. |
| `requireNetworkBandwidth` | `25000` | Schedules the Job onto a node with at least 25 Gbps of network bandwidth, so the HuggingFace-to-S3 transfer isn't network-bound. |
| `parameters.fileWorkers` | `30` | Downloads up to 30 files in parallel to maximize throughput. There are 141 safetensors files with a max size of ~5.5GB each, so `storageSize` must scale accordingly - see the [sizing note](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) in the model-cache guide. |

```bash
export MODEL_NAME="zai-org/GLM-5.3"
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
    fileWorkers: 30
EOF
```

> Total time: 413s

## Deploy Model

```bash
export IMAGE="<IMAGE>"
envsubst '${IMAGE} ${S3_BUCKET} ${INFERENCE_SA}' < vllm/b300-agg-mx-s3.yaml | kubectl apply -n dynamo-system -f -

# Watch all logs
kubectl -n dynamo-system logs -l  nvidia.com/dynamo-graph-deployment-name=glm53-b300-agg-mx-s3-vllm --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f

# Watch first worker logs (target)
kubectl -n dynamo-system logs -l  nvidia.com/dynamo-graph-deployment-name=glm53-b300-agg-mx-s3-vllm,nvidia.com/dynamo-component=VllmWorker,grove.io/podclique-pod-index=0 --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f

# Watch 2nd worker logs (source)
kubectl -n dynamo-system logs -l  nvidia.com/dynamo-graph-deployment-name=glm53-b300-agg-mx-s3-vllm,nvidia.com/dynamo-component=VllmWorker,grove.io/podclique-pod-index=1 --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f

# Delete
envsubst '${IMAGE} ${S3_BUCKET} ${INFERENCE_SA}' < vllm/b300-agg-mx-s3.yaml | kubectl delete -n dynamo-system -f -
```

## Scale replicas

```bash
kubectl patch dynamographdeployment glm53-b300-agg-mx-s3-vllm \
  -n dynamo-system \
  --type='json' \
  -p='[
    {"op": "test", "path": "/spec/components/2/name", "value": "VllmWorker"},
    {"op": "replace", "path": "/spec/components/2/replicas", "value": 2}
  ]'
```
