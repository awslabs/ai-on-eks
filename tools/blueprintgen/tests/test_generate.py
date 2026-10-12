# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import yaml

from tools.blueprintgen.generate import generate

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _load(name):
    with open(os.path.join(REPO_ROOT, "registry", "models", name)) as fh:
        return yaml.safe_load(fh)


def test_vllm_manifest_parses_and_has_expected_fields():
    entry = _load("qwen3-coder-30b.yaml")
    manifest = generate(entry, target="vllm")
    docs = list(yaml.safe_load_all(manifest))
    kinds = {d["kind"] for d in docs}
    assert kinds == {"Deployment", "Service"}

    deployment = next(d for d in docs if d["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    args = container["args"]
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in args
    assert "--tensor-parallel-size" in args
    assert args[args.index("--tensor-parallel-size") + 1] == "4"
    assert args[args.index("--max-model-len") + 1] == "262144"
    # tool parser flags present because entry has tool_parser
    assert "--enable-auto-tool-choice" in args
    assert args[args.index("--tool-call-parser") + 1] == "qwen3_coder"
    # HF token sourced from k8s secret hf-token
    env = container["env"]
    hf = next(e for e in env if e["name"] == "HF_TOKEN")
    assert hf["valueFrom"]["secretKeyRef"]["name"] == "hf-token"
    # karpenter nodeSelector by instance type
    node_selector = deployment["spec"]["template"]["spec"]["nodeSelector"]
    assert node_selector["node.kubernetes.io/instance-type"] == "g6e.12xlarge"


def test_dynamo_manifest_is_graph_deployment():
    entry = _load("qwen3-coder-30b.yaml")
    manifest = generate(entry, target="dynamo")
    doc = yaml.safe_load(manifest)
    assert doc["kind"] == "DynamoGraphDeployment"
    services = doc["spec"]["services"]
    assert "Frontend" in services
    worker = services["VllmWorker"]
    args = worker["extraPodSpec"]["mainContainer"]["args"]
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in args
    assert args[args.index("--tensor-parallel-size") + 1] == "4"


def test_no_tool_parser_flags_when_unset():
    entry = _load("deepseek-r1-distill-llama-70b.yaml")  # has no tool_parser
    # unverified entries lack instances; supply an instance type for nodeSelector
    entry = dict(entry)
    entry["instances"] = [{"type": "g6e.48xlarge", "region_tested": "us-east-2"}]
    manifest = generate(entry, target="vllm")
    docs = list(yaml.safe_load_all(manifest))
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    args = deployment["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--tool-call-parser" not in args
    assert "--enable-auto-tool-choice" not in args


def test_invalid_entry_raises_value_error():
    with pytest.raises(ValueError):
        generate({"name": "broken"}, target="vllm")


def test_unknown_target_raises_value_error():
    entry = _load("qwen3-coder-30b.yaml")
    with pytest.raises(ValueError):
        generate(entry, target="tensorrt")
