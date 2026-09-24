# Model Registry

Each `models/<name>.yaml` describes one open-weight model verified (or in progress)
for deployment on EKS: architecture, Hugging Face repo, tensor-parallel size,
verified instance types, precision, benchmark results, and status.

## Fields

See `schema.json` (JSON Schema draft 2020-12). `status` is one of `verified`,
`unverified`, or `draft`. A `verified` entry MUST carry `verified_date`, at least one
`instances` benchmark row, and a `guide_url`. Facts are verified-or-unset:
`tool_parser` appears only when confirmed against the vLLM arch→parser table.

## Validate

    pip install -e .[test]
    python -m registry.validate      # validates every models/*.yaml

CI runs the same command; a non-zero exit fails the build.
