#!/bin/bash
# Agent Sandbox — Basic blueprint apply.
#
# Smallest viable Sandbox deployment, no addons. Auto-detects the
# cluster's compute mode and claims the matching basic SandboxTemplate
# (sandbox-runc on Auto Mode, sandbox-gvisor on Standard EKS).
# Default workload is nginx:alpine baked into the basic template — the
# K8s shell-demo image, picked for familiarity.
#
# Use this as the first tier of testing after the platform infra is
# up, before layering on the reference agent, KRO, or egress.
#
# Prerequisites:
#   - infra/agent-sandbox provisioned + platform manifests applied
#     (namespace, RuntimeClass for gVisor on Standard EKS, both
#     SandboxTemplates).
#   - kubectl configured against the cluster.
#
# Usage:
#   cd blueprints/agent-sandbox/basic
#   ./install.sh                # Mode detection + apply + wait for Ready
#   ./install.sh smoke          # Apply + smoke test (kubectl exec into the pod)
#   ./install.sh uninstall      # Remove the SandboxClaim

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_DIR="$(cd "$SCRIPT_DIR/../../../infra/agent-sandbox" && pwd)"
NS="agent-sandboxes"
CLAIM_NAME="sandbox-basic"
PHASE="${1:-apply}"

# Region precedence (matches conformance.sh / cleanup.sh):
#   tfvars > AWS_REGION env > AWS_DEFAULT_REGION env > kubectl context
#   > base module default (us-west-2).
TFVARS_REGION=""
if [ -f "$INFRA_DIR/terraform/blueprint.tfvars" ]; then
    TFVARS_REGION=$(grep -E '^region\s*=' "$INFRA_DIR/terraform/blueprint.tfvars" \
        | head -1 | awk -F'"' '{print $2}' || echo "")
fi
if [ -n "$TFVARS_REGION" ]; then
    REGION="$TFVARS_REGION"
elif [ -n "${AWS_REGION:-}" ]; then
    REGION="$AWS_REGION"
elif [ -n "${AWS_DEFAULT_REGION:-}" ]; then
    REGION="$AWS_DEFAULT_REGION"
else
    REGION=$(kubectl config current-context 2>/dev/null \
        | awk -F':' '{print $4}' || echo "")
    REGION="${REGION:-us-west-2}"
fi
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"

detect_compute_mode() {
    # Same shape as conformance.sh — query the AWS API for compute mode
    # with a small retry loop to absorb transient failures.
    local enabled attempt
    for attempt in 1 2 3; do
        enabled=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
            --query 'cluster.computeConfig.enabled' --output text 2>/dev/null || echo "")
        if [ "$enabled" = "True" ] || [ "$enabled" = "true" ]; then
            COMPUTE_MODE="automode"
            SANDBOX_TEMPLATE="sandbox-runc"
            echo "=== EKS Auto Mode detected — claiming SandboxTemplate '$SANDBOX_TEMPLATE' ==="
            return 0
        fi
        if [ "$enabled" = "False" ] || [ "$enabled" = "false" ]; then
            COMPUTE_MODE="standard"
            SANDBOX_TEMPLATE="sandbox-gvisor"
            echo "=== Standard EKS detected — claiming SandboxTemplate '$SANDBOX_TEMPLATE' ==="
            return 0
        fi
        if [ "$attempt" -lt 3 ]; then sleep 2; fi
    done
    echo "FAIL: Could not determine cluster compute mode after 3 attempts." >&2
    exit 1
}

apply_claim() {
    detect_compute_mode

    # Make sure the platform manifests are in place before we claim. If
    # the namespace is missing, the user hasn't applied the infra
    # manifests yet — fail fast with a pointer.
    kubectl get ns "$NS" >/dev/null 2>&1 || {
        echo "FAIL: Namespace '$NS' missing. Apply $INFRA_DIR/manifests/namespace.yaml first." >&2
        exit 1
    }
    kubectl -n "$NS" get sandboxtemplate "$SANDBOX_TEMPLATE" >/dev/null 2>&1 || {
        echo "FAIL: SandboxTemplate '$SANDBOX_TEMPLATE' missing. Apply $INFRA_DIR/manifests/sandbox-{runc,gvisor}.yaml first." >&2
        exit 1
    }

    echo "=== Applying basic SandboxClaim ==="
    sed "s|__SANDBOX_TEMPLATE__|$SANDBOX_TEMPLATE|g" \
        "$SCRIPT_DIR/sandbox-claim-basic.yaml" \
        | kubectl apply -f -

    echo "=== Waiting for Sandbox pod Ready (up to 3 min) ==="
    # The pod's `metadata.name` is the same as the SandboxClaim's name.
    kubectl -n "$NS" wait --for=condition=Ready pod/"$CLAIM_NAME" --timeout=180s

    echo ""
    echo "=== Basic sandbox ready ==="
    echo ""
    echo "Compute mode:    $COMPUTE_MODE"
    echo "Sandbox template: $SANDBOX_TEMPLATE"
    echo "Default image:   nginx:alpine"
    echo ""
    echo "Verify the workload:"
    echo "  kubectl exec -n $NS $CLAIM_NAME -- nginx -v"
    echo ""
    echo "Or run the smoke test:"
    echo "  ./install.sh smoke"
    echo ""
    echo "Customize: copy sandbox-claim-basic.yaml + write your own SandboxTemplate"
    echo "with a different image. The Sandbox shape (security context, egress"
    echo "labels, runtime class) carries over to any workload."
}

smoke_test() {
    apply_claim
    echo ""
    echo "=== Smoke test ==="
    # nginx -v writes the version to stderr. Redirect so the assertion
    # below picks it up cleanly.
    if kubectl exec -n "$NS" "$CLAIM_NAME" -- nginx -v 2>&1 | grep -q "nginx version"; then
        echo "PASS: nginx is running inside the sandbox."
    else
        echo "FAIL: nginx version probe didn't return expected output." >&2
        exit 1
    fi
}

uninstall() {
    detect_compute_mode || true
    echo "=== Removing basic SandboxClaim ==="
    if [ -n "${SANDBOX_TEMPLATE:-}" ]; then
        sed "s|__SANDBOX_TEMPLATE__|$SANDBOX_TEMPLATE|g" \
            "$SCRIPT_DIR/sandbox-claim-basic.yaml" \
            | kubectl delete -f - --ignore-not-found
    else
        # Fall back to deleting by name+namespace if mode detection
        # failed (e.g., cluster already gone).
        kubectl -n "$NS" delete sandboxclaim "$CLAIM_NAME" --ignore-not-found
    fi
    echo "=== Basic SandboxClaim removed ==="
}

case "$PHASE" in
    apply)
        apply_claim
        ;;
    smoke)
        smoke_test
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Unknown phase: $PHASE" >&2
        echo "Usage: $0 [apply|smoke|uninstall]" >&2
        exit 1
        ;;
esac
