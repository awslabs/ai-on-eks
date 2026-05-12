#!/bin/bash
# Agent Sandbox blueprint — end-to-end conformance test.
#
# Validates the full chain that the blueprint installs:
#   - agent-sandbox controller resolves a Sandbox resource
#   - Karpenter provisions a gVisor-runtime node on demand
#   - IRSA injects Bedrock credentials into the sandbox pod
#   - Cilium FQDN allowlist permits pypi.org + bedrock-runtime + sts
#   - Cilium FQDN allowlist blocks a non-allowlisted domain
#
# Run after `infra/agent-sandbox/install.sh` completes successfully.
# Exits 0 on pass, 1 on any failure. No interactive prompts.
#
# Usage:
#   CLUSTER_NAME=agent-sandbox \
#   BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
#     ./conformance.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_DIR="$SCRIPT_DIR/../../../infra/agent-sandbox"
NS="agent-sandboxes"
SA="sandbox-agent-sa"
POD="sandbox-agent"
CONFIGMAP="sandbox-agent-script"
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"
REGION="${AWS_REGION:-us-east-1}"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

log() {
    echo "[$(date +%H:%M:%S)] $*"
}

require_env() {
    if [ -z "${BEDROCK_ROLE_ARN:-}" ]; then
        fail "BEDROCK_ROLE_ARN not set. Export the IAM role ARN that grants bedrock:InvokeModel on the target model, then re-run."
    fi
}

require_cluster() {
    log "Checking cluster reachability + prerequisites..."
    kubectl cluster-info >/dev/null 2>&1 || fail "kubectl cannot reach the cluster; run 'aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION'"
    kubectl get ns "$NS" >/dev/null 2>&1 || fail "Namespace '$NS' missing; run 'infra/agent-sandbox/install.sh manifests'"
    kubectl get runtimeclass gvisor >/dev/null 2>&1 || fail "RuntimeClass 'gvisor' missing; run 'infra/agent-sandbox/install.sh manifests'"
    kubectl get sandboxtemplate sandbox-gvisor -n "$NS" >/dev/null 2>&1 || fail "SandboxTemplate 'sandbox-gvisor' missing; run 'infra/agent-sandbox/install.sh manifests'"
    kubectl -n agent-sandbox-system get deployment agent-sandbox-controller >/dev/null 2>&1 || fail "agent-sandbox controller missing; run 'infra/agent-sandbox/install.sh sandbox'"
}

setup_configmap_with_real_agent() {
    # sandbox-agent.yaml embeds a ConfigMap with a placeholder
    # agent.py. We apply sandbox-agent.yaml first (which creates the
    # SA + placeholder ConfigMap + Sandbox), then overwrite the
    # ConfigMap with the real agent.py contents, then bounce the pod
    # so the container's startup `cp /config/agent.py
    # /workspace/agent.py` picks up the real content. This order
    # avoids races where the placeholder content is copied into
    # /workspace and sticks there.
    log "Applying Sandbox manifest (creates SA + placeholder ConfigMap + Sandbox)..."
    kubectl apply -f "$INFRA_DIR/manifests/sandbox-agent.yaml" >/dev/null

    log "Replacing placeholder ConfigMap with real agent.py contents..."
    kubectl -n "$NS" create configmap "$CONFIGMAP" \
        --from-file=agent.py="$SCRIPT_DIR/agent.py" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null

    # Force pod recreation so the container's one-shot `cp` at
    # startup reads the real ConfigMap content.
    log "Recreating Sandbox pod so it mounts the real agent.py..."
    kubectl -n "$NS" delete pod "$POD" --ignore-not-found --wait=true >/dev/null 2>&1 || true
}

setup_irsa_annotation() {
    # IRSA wiring — annotate the ServiceAccount with the IAM role ARN
    # so the EKS admission controller injects AWS_WEB_IDENTITY_TOKEN_FILE
    # + AWS_ROLE_ARN into the Sandbox pod.
    #
    # gVisor's Sentry network namespace doesn't forward the link-local
    # route to 169.254.170.23, so EKS Pod Identity doesn't work for
    # sandboxes on the gVisor tier. IRSA routes through STS over the
    # regular network path (covered by the Cilium FQDN allowlist) and
    # works transparently.
    #
    # Environment variables expected:
    #   BEDROCK_ROLE_ARN  — IAM role ARN with bedrock:InvokeModel +
    #                       IRSA trust policy allowing the cluster's
    #                       OIDC provider for
    #                       system:serviceaccount:agent-sandboxes:sandbox-agent-sa
    log "Ensuring ServiceAccount $SA exists + has IRSA annotation..."
    if ! kubectl -n "$NS" get serviceaccount "$SA" >/dev/null 2>&1; then
        kubectl -n "$NS" create serviceaccount "$SA" >/dev/null
    fi
    kubectl annotate serviceaccount "$SA" -n "$NS" \
        "eks.amazonaws.com/role-arn=$BEDROCK_ROLE_ARN" \
        --overwrite >/dev/null
}

wait_for_pod() {
    log "Waiting for Sandbox pod Ready (up to 5 min)..."
    # Sandbox controller recreates the pod after our delete; give it
    # a moment to spawn a fresh one before waiting on Ready.
    sleep 5
    if ! kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=300s >/dev/null; then
        kubectl -n "$NS" describe "pod/$POD" >&2
        fail "Sandbox pod did not become Ready within 5 min"
    fi
    log "Pod Ready."
}

assert_runtime_class() {
    log "Asserting pod is scheduled with runtimeClassName=gvisor..."
    local rc
    rc=$(kubectl -n "$NS" get "pod/$POD" -o jsonpath='{.spec.runtimeClassName}')
    [ "$rc" = "gvisor" ] || fail "Expected runtimeClassName=gvisor, got '$rc'"
}

assert_policies_valid() {
    log "Asserting Cilium policies are Valid..."
    local admin_valid app_valid
    admin_valid=$(kubectl get ciliumclusterwidenetworkpolicy admin-block-imds -o jsonpath='{.status.conditions[?(@.type=="Valid")].status}' 2>/dev/null || echo "")
    app_valid=$(kubectl -n "$NS" get ciliumnetworkpolicy sandbox-llm-allowlist -o jsonpath='{.status.conditions[?(@.type=="Valid")].status}' 2>/dev/null || echo "")
    [ "$admin_valid" = "True" ] || fail "CiliumClusterwideNetworkPolicy admin-block-imds not Valid (got '$admin_valid')"
    [ "$app_valid" = "True" ] || fail "CiliumNetworkPolicy sandbox-llm-allowlist not Valid (got '$app_valid')"
}

run_agent_and_validate() {
    log "Running agent.py inside the sandbox..."
    local output
    output=$(kubectl exec -n "$NS" "$POD" -c agent-runtime -- python /workspace/agent.py 2>&1) || {
        echo "$output" >&2
        fail "agent.py exited non-zero"
    }
    echo "$output"
    echo "---"
    log "Validating expected PASS / BLOCKED markers..."
    echo "$output" | grep -q "PASS: boto3 installed" || fail "Step 1 (PyPI install) did not PASS"
    echo "$output" | grep -q "Bedrock reply" || fail "Step 2 (Bedrock call) did not return a reply"
    echo "$output" | grep -q "PASS: snippet exited 0" || fail "Step 3 (snippet execution) did not PASS"
    echo "$output" | grep -qE "BLOCKED: https://blocked-example\.example\.com" || fail "Step 4 (FQDN block) did not BLOCK"
    echo "$output" | grep -qE "BLOCKED: 8\.8\.8\.8:443" || fail "Step 5 (IP block) did not BLOCK"
    log "All 5 expected outcomes matched."
}

cleanup() {
    # Default: leave the sandbox running so repeat conformance runs
    # are fast (no re-provisioning gVisor nodes). Pass CLEANUP=1 to
    # tear down the sandbox resources on exit.
    if [ "${CLEANUP:-0}" != "1" ]; then
        log "Leaving Sandbox + ConfigMap in place (set CLEANUP=1 to remove)."
        return 0
    fi
    log "Removing test-run resources (Sandbox pod + ConfigMap)..."
    kubectl delete -f "$INFRA_DIR/manifests/sandbox-agent.yaml" --ignore-not-found >/dev/null 2>&1 || true
    kubectl -n "$NS" delete configmap "$CONFIGMAP" --ignore-not-found >/dev/null 2>&1 || true
    log "Cleanup complete. IAM role + IRSA annotation retained for re-runs."
}

main() {
    trap cleanup EXIT
    require_env
    require_cluster
    setup_irsa_annotation
    setup_configmap_with_real_agent
    wait_for_pod
    assert_runtime_class
    assert_policies_valid
    run_agent_and_validate
    log ""
    log "PASS: blueprint conformance test succeeded."
}

main "$@"
