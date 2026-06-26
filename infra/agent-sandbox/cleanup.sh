#!/bin/bash
# Agent Sandbox — teardown wrapper.
#
# Thin wrapper over the shared cleanup driver (infra/base/cleanup/run-cleanup.sh).
# Resolves cluster + region from the live tfvars and passes the agent-sandbox
# egress-example uninstall as the component phase-0 hook. The phased teardown
# logic (Karpenter scale-down/finalizer/instance sweep, retry-and-verify base
# destroy, auxiliary resource sweep) lives in the shared driver so every
# blueprint reuses it — see issue #334.
#
# Usage:
#   cd infra/agent-sandbox
#   ./cleanup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$SCRIPT_DIR/terraform/_LOCAL"
DRIVER="$(cd "$SCRIPT_DIR/../base/cleanup" && pwd)/run-cleanup.sh"

# Resolve cluster + region from tfvars (region precedence:
# tfvars > AWS_REGION > AWS_DEFAULT_REGION > kubectl context > us-west-2).
if [ -f "$SCRIPT_DIR/terraform/blueprint.tfvars" ]; then
    CLUSTER_NAME=$(grep -E '^name\s*=' "$SCRIPT_DIR/terraform/blueprint.tfvars" | head -1 | awk -F'"' '{print $2}')
    TFVARS_REGION=$(grep -E '^region\s*=' "$SCRIPT_DIR/terraform/blueprint.tfvars" | head -1 | awk -F'"' '{print $2}' || echo "")
fi
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"
if [ -n "${TFVARS_REGION:-}" ]; then
    REGION="$TFVARS_REGION"
elif [ -n "${AWS_REGION:-}" ]; then
    REGION="$AWS_REGION"
elif [ -n "${AWS_DEFAULT_REGION:-}" ]; then
    REGION="$AWS_DEFAULT_REGION"
else
    REGION=$(kubectl config current-context 2>/dev/null | awk -F':' '{print $4}' || echo "")
    REGION="${REGION:-us-west-2}"
fi

# Phase-0 hook: the agent-egress example's idempotent, mode-aware uninstall
# (releases CNPs/ANPs + the Bedrock IRSA role it provisioned).
EGRESS_DIR="$SCRIPT_DIR/../../blueprints/agent-sandbox/egress"
PHASE0_HOOK=""
if [ -x "$EGRESS_DIR/install.sh" ]; then
    PHASE0_HOOK="cd '$EGRESS_DIR' && ./install.sh uninstall"
fi

CLUSTER_NAME="$CLUSTER_NAME" REGION="$REGION" LOCAL_DIR="$LOCAL_DIR" PHASE0_HOOK="$PHASE0_HOOK" \
    bash "$DRIVER"
