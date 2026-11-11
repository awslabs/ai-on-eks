# nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16

Model: [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16)

> **Note:** Run all commands in this recipe from the recipe folder (`blueprints/inference/nvidia-dynamo/recipes/nemotron-3.5-bolt-e4b`). Manifest paths below are relative to that directory. If you are coming from the main setup guide, switch into it first:
>
> ```bash
> cd recipes/nemotron-3.5-bolt-e4b
> ```

## Sync model weights

Follow the [model-cache](../../model-cache/README.md) guide to sync the nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 weights from HuggingFace to either a `model-cache` PVC or an Amazon S3 bucket. Both workflows are documented there.

For this model, the recommended path is [Amazon S3 bucket](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3).

When running the [HuggingFace-to-S3](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) sync job, use the following values in the `s3ModelCopy` configuration. These are tuned for the size of nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16:

| Parameter | Recommended value | Why |
| --- | --- | --- |
| `storageSize` | `50` | Scratch volume in GiB for the download. |
| `requireNetworkBandwidth` | `5000` | Schedules the Job onto a node with at least 25 Gbps of network bandwidth, so the HuggingFace-to-S3 transfer isn't network-bound. |
| `parameters.fileWorkers` | `2` | Downloads up to 35 files in parallel to maximize throughput. There are 1 safetensor file that is about ~16GB in size, so `storageSize` must scale accordingly - see the [sizing note](../../model-cache/README.md#sync-model-weights-from-huggingface-to-amazon-s3) in the model-cache guide. |

```bash
export MODEL_NAME="nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16"
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
  storageSize: 50
  requireNetworkBandwidth: 5000
  parameters:
    fileWorkers: 2
EOF
```

> Total time: 28s

## Deploy Model - vLLM

```bash
envsubst '${S3_BUCKET} ${INFERENCE_SA}' < vllm/g6e-agg-s3.yaml | kubectl apply -n dynamo-system -f -

# Watch all logs
kubectl -n dynamo-system logs -l nvidia.com/dynamo-graph-deployment-name=nemotron-35-bolt-e4b-agg-vllm --all-containers=true --max-log-requests=20 --prefix=true --timestamps -f
```

## Smoke test

Wait for the worker to become ready, then port-forward the frontend service to your machine:

```bash
kubectl -n dynamo-system wait pod \
  -l nvidia.com/dynamo-graph-deployment-name=nemotron-35-bolt-e4b-agg-vllm \
  --for=condition=Ready --timeout=20m

kubectl -n dynamo-system port-forward svc/nemotron-35-bolt-e4b-agg-vllm-frontend 8000:8000
```

In a second terminal, confirm the model is registered and send a chat completion request:

```bash
curl -s localhost:8000/v1/models | jq

curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16",
    "messages": [
      {"role": "user", "content": "Explain what Kubernetes is in two sentences."}
    ],
    "max_tokens": 256,
    "stream": false
  }' | jq
```

A successful response returns a `choices[0].message.content` field with the model's answer. Stop the port-forward with `Ctrl+C` when you're done.

## Cleanup

```bash
envsubst '${S3_BUCKET} ${INFERENCE_SA}' < vllm/g6e-agg-s3.yaml | kubectl delete -n dynamo-system -f -
```
