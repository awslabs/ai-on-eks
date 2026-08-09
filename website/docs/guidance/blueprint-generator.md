---
title: Blueprint Generator
sidebar_label: Blueprint Generator
---

# Blueprint Generator

`aoe-blueprint` converts a registry entry into a ready-to-apply Kubernetes manifest,
so you never hand-write vLLM args or GPU node selectors.

## Usage

    aoe-blueprint gen registry/models/qwen3-coder-30b.yaml --target vllm -o qwen3.yaml
    kubectl apply -f qwen3.yaml

`--target vllm` produces a vLLM `Deployment` and `Service`. `--target dynamo` produces
an NVIDIA `DynamoGraphDeployment` CRD (Frontend plus worker). Both:

- source the Hugging Face token from the `hf-token` Kubernetes secret,
- pin the pod to the verified instance type via a Karpenter node selector,
- pass `--model`, `--tensor-parallel-size`, and `--max-model-len` from the entry,
- add tool-call parser flags only when the entry declares a verified `tool_parser`.

The entry is validated against the registry schema before anything is rendered.
