# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry entry -> deployable Kubernetes manifest YAML."""
from __future__ import annotations

import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from registry.validate import load_schema, validate_entry

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
_REGISTRY_SCHEMA = os.path.join(
    os.path.dirname(__file__), "..", "..", "registry", "schema.json"
)
_TEMPLATES = {"vllm": "vllm.yaml.j2", "dynamo": "dynamo.yaml.j2"}


def _instance_type(entry: dict) -> str:
    instances = entry.get("instances") or []
    if not instances:
        raise ValueError(
            "entry has no instances; supply at least one instance type for nodeSelector"
        )
    return instances[0]["type"]


def generate(entry: dict, target: str = "vllm") -> str:
    """Registry entry dict -> Kubernetes manifest YAML string.

    target: "vllm" (Deployment+Service) | "dynamo" (DynamoGraphDeployment CRD).
    Raises ValueError if the entry fails schema validation or target is unknown.
    """
    if target not in _TEMPLATES:
        raise ValueError(f"unknown target {target!r}; expected one of {sorted(_TEMPLATES)}")

    schema = load_schema(os.path.abspath(_REGISTRY_SCHEMA))
    errors = validate_entry(entry, schema)
    if errors:
        raise ValueError("entry failed schema validation: " + "; ".join(errors))

    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATES[target])
    return template.render(
        name=entry["name"],
        served_repo=entry.get("hf_repo_serve") or entry["hf_repo"],
        tensor_parallel=entry["tensor_parallel"],
        max_model_len=entry["max_model_len"],
        tool_parser=entry.get("tool_parser"),
        container_image=entry["container_image"],
        instance_type=_instance_type(entry),
    )
