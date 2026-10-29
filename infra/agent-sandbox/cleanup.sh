#!/bin/bash
# Agent Sandbox teardown. Two steps:
#   1. Remove resources this blueprint created outside Terraform
#      (in-cluster CRs, namespace, RuntimeClasses, egress example).
#   2. Run the standard Terraform cleanup from the base copy.
#
# Usage:
#   cd infra/agent-sandbox
#   ./cleanup.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$SCRIPT_DIR/terraform/_LOCAL"

echo "=== Step 1: blueprint-created resources ==="

# Egress example uninstall (idempotent, releases CNPs/ANPs + its IRSA role).
EGRESS_DIR="$SCRIPT_DIR/../../blueprints/agent-sandbox/egress"
if [ -x "$EGRESS_DIR/install.sh" ]; then
    (cd "$EGRESS_DIR" && ./install.sh uninstall) || true
fi

if kubectl get namespace agent-sandboxes >/dev/null 2>&1; then
    # Delete Microvm/MicrovmImage CRs first and wait: the ACK controller
    # reaps the service-side VMs and images through its finalizers. If
    # the controller is destroyed before these CRs are gone, the Lambda
    # resources are orphaned in the account.
    kubectl delete microvms,microvmimages -n agent-sandboxes --all --timeout=300s 2>/dev/null || true

    # Session and capacity resources, then the namespace.
    kubectl delete sandboxclaims,sandboxes,sandboxwarmpools,sandboxtemplates \
        -n agent-sandboxes --all --timeout=120s 2>/dev/null || true
    kubectl delete namespace agent-sandboxes --timeout=120s || true
fi

# Cluster-scoped RuntimeClasses applied from manifests/.
kubectl delete runtimeclass gvisor kata-fc --ignore-not-found 2>/dev/null || true

echo "=== Step 2: Terraform cleanup ==="
cd "$LOCAL_DIR"
source ./cleanup.sh
