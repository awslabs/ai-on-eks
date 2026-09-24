# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os

import pytest

from registry.validate import load_schema, validate_entry, validate_all

HERE = os.path.dirname(__file__)
FIXTURES = os.path.join(HERE, "fixtures")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def _load_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def test_valid_minimal_entry_passes():
    schema = load_schema(os.path.join(REPO_ROOT, "registry", "schema.json"))
    entry = _load_yaml(os.path.join(FIXTURES, "valid_min.yaml"))
    assert validate_entry(entry, schema) == []


def test_missing_required_field_reports_error():
    schema = load_schema(os.path.join(REPO_ROOT, "registry", "schema.json"))
    entry = _load_yaml(os.path.join(FIXTURES, "invalid_missing_required.yaml"))
    errors = validate_entry(entry, schema)
    assert any("hf_repo" in e for e in errors)


def test_bad_status_enum_reports_error():
    schema = load_schema(os.path.join(REPO_ROOT, "registry", "schema.json"))
    entry = _load_yaml(os.path.join(FIXTURES, "invalid_bad_status.yaml"))
    errors = validate_entry(entry, schema)
    assert any("status" in e for e in errors)


def test_verified_entry_requires_instances_and_verified_date():
    schema = load_schema(os.path.join(REPO_ROOT, "registry", "schema.json"))
    # verified but no instances / verified_date -> must fail
    entry = {
        "name": "x",
        "hf_repo": "org/X",
        "architecture": "Qwen3MoeForCausalLM",
        "params_b": 30,
        "precision": ["bf16"],
        "tensor_parallel": 4,
        "max_model_len": 262144,
        "container_image": "pending",
        "status": "verified",
    }
    errors = validate_entry(entry, schema)
    assert errors != []


def test_all_seed_entries_are_valid():
    results = validate_all(
        models_dir=os.path.join(REPO_ROOT, "registry", "models"),
        schema_path=os.path.join(REPO_ROOT, "registry", "schema.json"),
    )
    bad = {k: v for k, v in results.items() if v}
    assert bad == {}, f"seed entries failed validation: {bad}"
