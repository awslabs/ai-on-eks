# Blueprint Generator (`aoe-blueprint`)

Turns a validated registry entry into a deployable Kubernetes manifest.

## Install

    pip install -e ../../registry   # provides registry.validate for schema checks
    pip install -e .

## Use

    aoe-blueprint gen registry/models/qwen3-coder-30b.yaml --target vllm -o qwen3.yaml
    aoe-blueprint gen registry/models/qwen3-coder-30b.yaml --target dynamo

`--target vllm` emits a plain vLLM `Deployment` + `Service`. `--target dynamo` emits an
NVIDIA `DynamoGraphDeployment` CRD with a `Frontend` and a worker. Both source the HF
token from the `hf-token` Kubernetes secret, pin the node with a Karpenter
`node.kubernetes.io/instance-type` selector, and pass `--model`, `--tensor-parallel-size`,
and `--max-model-len` to `vllm serve`. Tool-call parser flags are added only when the
entry declares a verified `tool_parser`.

The entry is validated against `registry/schema.json` before rendering; invalid entries
raise an error.
