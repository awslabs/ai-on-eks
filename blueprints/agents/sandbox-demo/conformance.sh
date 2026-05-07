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
SA="sandbox-demo-agent"
POD="sandbox-demo"
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

setup_configmap() {
    log "Installing agent.py into ConfigMap..."
    kubectl -n "$NS" delete configmap sandbox-demo-agent-script --ignore-not-found >/dev/null
    kubectl -n "$NS" create configmap sandbox-demo-agent-script \
        --from-file=agent.py="$SCRIPT_DIR/agent.py" >/dev/null
}

setup_pod_identity() {
    log "Checking Pod Identity association for ServiceAccount $SA..."
    local existing
    existing=$(aws eks list-pod-identity-associations \
        --cluster-name "$CLUSTER_NAME" \
        --namespace "$NS" \
        --service-account "$SA" \
        --region "$REGION" \
        --query 'associations[0].associationId' \
        --output text 2>/dev/null || true)
    if [ -n "$existing" ] && [ "$existing" != "None" ]; then
        log "Pod Identity association already exists: $existing"
        return 0
    fi
    log "Creating Pod Identity association..."
    aws eks create-pod-identity-association \
        --cluster-name "$CLUSTER_NAME" \
        --namespace "$NS" \
        --service-account "$SA" \
        --role-arn "$BEDROCK_ROLE_ARN" \
        --region "$REGION" >/dev/null
}

wait_for_pod() {
    log "Applying Sandbox + waiting for pod Ready (up to 5 min)..."
    kubectl apply -f "$INFRA_DIR/manifests/demo-agent.yaml" >/dev/null
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
    echo "$output" | grep -qE "BLOCKED: https://demo-blocked\.example\.com" || fail "Step 4 (blocked egress) did not BLOCK"
    log "All 4 expected outcomes matched."
}

cleanup() {
    log "Removing test-run resources (Sandbox pod + ConfigMap)..."
    kubectl delete -f "$INFRA_DIR/manifests/demo-agent.yaml" --ignore-not-found >/dev/null 2>&1 || true
    kubectl -n "$NS" delete configmap sandbox-demo-agent-script --ignore-not-found >/dev/null 2>&1 || true
    log "Cleanup complete. Pod Identity association + IAM role retained for re-runs."
}

main() {
    trap cleanup EXIT
    require_env
    require_cluster
    setup_configmap
    setup_pod_identity
    wait_for_pod
    assert_runtime_class
    assert_policies_valid
    run_agent_and_validate
    log ""
    log "PASS: blueprint conformance test succeeded."
}

main "$@"
