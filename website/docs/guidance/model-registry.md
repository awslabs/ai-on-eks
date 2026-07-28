---
title: Model Registry
sidebar_label: Model Registry
---

# Model Registry

The model registry is the single source of truth for which open-weight models are
verified to run on EKS and how. Each entry (`registry/models/<name>.yaml`) records the
model's architecture, Hugging Face repo, tensor-parallel size, precision options,
verified instance types with benchmark numbers, and a status of `verified`,
`unverified`, or `draft`.

## Why it exists

Deploying a frontier open model correctly means knowing its architecture class, the
GPU count and instance type that fit it, the right `--max-model-len`, and whether a
tool-call parser is available. The registry captures that once, verified, so the
blueprint generator and the deploy skills can act on it without re-deriving it.

## Validation

Entries validate against `registry/schema.json` (JSON Schema draft 2020-12):

    python -m registry.validate

A `verified` entry must include `verified_date`, at least one benchmarked instance
row, and a `guide_url` pointing at its generated deployment guide.
