# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate ai-on-eks model registry entries against registry/schema.json."""
from __future__ import annotations

import glob
import json
import os
import sys

import yaml
from jsonschema import Draft202012Validator

DEFAULT_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.json")
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_schema(schema_path: str = DEFAULT_SCHEMA_PATH) -> dict:
    with open(schema_path) as fh:
        return json.load(fh)


def validate_entry(entry: dict, schema: dict) -> list[str]:
    """Return a list of human-readable error strings. Empty list means valid."""
    validator = Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(entry), key=lambda e: list(e.path)):
        location = "/".join(str(p) for p in err.path) or "<root>"
        errors.append(f"{location}: {err.message}")
    return errors


def validate_all(
    models_dir: str = DEFAULT_MODELS_DIR,
    schema_path: str = DEFAULT_SCHEMA_PATH,
) -> dict[str, list[str]]:
    """Validate every registry/models/*.yaml. Returns {filename: [errors]}."""
    schema = load_schema(schema_path)
    results: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(models_dir, "*.yaml"))):
        with open(path) as fh:
            entry = yaml.safe_load(fh)
        results[os.path.basename(path)] = validate_entry(entry, schema)
    return results


def main(argv: list[str] | None = None) -> int:
    results = validate_all()
    failed = False
    for filename, errors in results.items():
        if errors:
            failed = True
            print(f"FAIL {filename}")
            for e in errors:
                print(f"  - {e}")
        else:
            print(f"OK   {filename}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
