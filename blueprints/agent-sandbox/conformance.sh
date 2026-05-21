#!/bin/bash
# Agent Sandbox blueprint — end-to-end conformance test.
#
# Validates the full chain that the blueprint installs:
#   - agent-sandbox controller resolves the SandboxClaim against an
#     agent-shaped SandboxTemplate (sandbox-agent-runc on Auto Mode,
#     sandbox-agent-gvisor on Standard EKS — auto-detected).
#   - Compute provisions the right node tier (Karpenter gVisor on
#     Standard EKS, EKS-managed node on Auto Mode).
#   - IRSA injects Bedrock credentials into the sandbox pod.
#   - Egress allowlist permits pypi.org + bedrock-runtime + sts and
#     blocks a non-allowlisted FQDN and raw IP.
#
# Run after `infra/agent-sandbox/install.sh` and the egress example
# (`blueprints/agent-sandbox/egress/install.sh`) have completed.
# Exits 0 on pass, 1 on any failure. No interactive prompts.
#
# Usage:
#   CLUSTER_NAME=agent-sandbox \
#   BEDROCK_ROLE_ARN=arn:aws:iam::<account>:role/<role-with-bedrock-invokemodel> \
#     ./conformance.sh
#
# Region is auto-resolved (tfvars > AWS_REGION > AWS_DEFAULT_REGION >
# kubectl context > us-west-2 default). Set AWS_REGION to override.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# conformance.sh lives at blueprints/agent-sandbox/. The platform manifests
# (RuntimeClass, SandboxTemplates, namespace) live at infra/agent-sandbox/manifests/.
# The workload manifest (SandboxClaim sandbox-agent.yaml) lives next to this
# script under blueprints/agent-sandbox/manifests/.
AGENT_DIR="$SCRIPT_DIR"
BLUEPRINT_MANIFEST_DIR="$SCRIPT_DIR/manifests"
INFRA_DIR="$(cd "$SCRIPT_DIR/../../infra/agent-sandbox" && pwd)"
INFRA_MANIFEST_DIR="$INFRA_DIR/manifests"
NS="agent-sandboxes"
SA="sandbox-agent-sa"
POD="sandbox-agent"
CONFIGMAP="sandbox-agent-script"
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"

# Region precedence (matches infra/agent-sandbox/cleanup.sh):
#   tfvars > AWS_REGION env > AWS_DEFAULT_REGION env > kubectl context
#   > base module default (us-west-2). The blueprint runs in whichever
#   region was deployed; a hard-coded default is a silent
#   wrong-region failure waiting to happen.
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
    # kubectl context ARN format: arn:aws:eks:<region>:<account>:cluster/<name>
    REGION=$(kubectl config current-context 2>/dev/null \
        | awk -F':' '{print $4}' || echo "")
    REGION="${REGION:-us-west-2}"
fi
echo "[$(date +%H:%M:%S)] Resolved cluster=$CLUSTER_NAME region=$REGION"

# Compute-mode detection. Auto Mode clusters cannot run gVisor (no
# node-level hooks for installing the runsc containerd shim), so the
# SandboxClaim's templateRef and the runtime-class assertion vary by
# mode. detect_compute_mode() resolves both at startup.
COMPUTE_MODE=""
SANDBOX_TEMPLATE=""

# Rendered SandboxClaim manifest (templateRef substituted). Created in
# setup_configmap_with_real_agent and reused at cleanup time.
RENDERED_SANDBOX_MANIFEST=""

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

detect_compute_mode() {
    # Set COMPUTE_MODE + SANDBOX_TEMPLATE based on the cluster's compute
    # config. Retry up to 3 times to absorb transient API failures so
    # we don't false-positive into the wrong code path.
    local enabled attempt
    for attempt in 1 2 3; do
        enabled=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
            --query 'cluster.computeConfig.enabled' --output text 2>/dev/null || echo "")
        if [ "$enabled" = "True" ] || [ "$enabled" = "true" ]; then
            COMPUTE_MODE="automode"
            SANDBOX_TEMPLATE="sandbox-agent-runc"
            log "Detected EKS Auto Mode — claiming SandboxTemplate '$SANDBOX_TEMPLATE' (gVisor unavailable on Auto Mode)."
            return 0
        fi
        if [ "$enabled" = "False" ] || [ "$enabled" = "false" ]; then
            COMPUTE_MODE="standard"
            SANDBOX_TEMPLATE="sandbox-agent-gvisor"
            log "Detected Standard EKS — claiming SandboxTemplate '$SANDBOX_TEMPLATE'."
            return 0
        fi
        if [ "$attempt" -lt 3 ]; then sleep 2; fi
    done
    fail "Could not determine cluster compute mode after 3 attempts. Check 'aws eks describe-cluster --name $CLUSTER_NAME --region $REGION' connectivity."
}

require_cluster() {
    log "Checking cluster reachability + prerequisites..."
    kubectl cluster-info >/dev/null 2>&1 || fail "kubectl cannot reach the cluster; run 'aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION'"
    kubectl get ns "$NS" >/dev/null 2>&1 || fail "Namespace '$NS' missing; apply $INFRA_MANIFEST_DIR/namespace.yaml first"
    kubectl -n agent-sandbox-system get deployment agent-sandbox-controller >/dev/null 2>&1 || fail "agent-sandbox controller missing; set enable_agent_sandbox=true in blueprint.tfvars and re-run install.sh"

    # SandboxTemplate for the chosen tier. The agent-shaped templates
    # ship with the blueprint (sandbox-agent-{runc,gvisor}.yaml). If the
    # caller hasn't applied them yet, do it now — keeps conformance
    # idempotent (the templates are stable, so reapply is a no-op).
    # Tier-specific cluster prerequisites (RuntimeClass, NodePool) come
    # from the platform infra and are only relevant for the gVisor tier
    # on Standard EKS.
    if ! kubectl -n "$NS" get sandboxtemplate "$SANDBOX_TEMPLATE" >/dev/null 2>&1; then
        # SANDBOX_TEMPLATE is `sandbox-agent-runc` (Auto Mode) or
        # `sandbox-agent-gvisor` (Standard EKS). The matching file uses
        # the same name with .yaml suffix.
        local agent_template_file="$BLUEPRINT_MANIFEST_DIR/${SANDBOX_TEMPLATE}.yaml"
        if [ -f "$agent_template_file" ]; then
            log "Applying agent SandboxTemplate from $agent_template_file..."
            kubectl apply -f "$agent_template_file" >/dev/null
        else
            fail "SandboxTemplate '$SANDBOX_TEMPLATE' missing and template file not found at $agent_template_file"
        fi
    fi

    if [ "$COMPUTE_MODE" = "standard" ]; then
        kubectl get runtimeclass gvisor >/dev/null 2>&1 \
            || fail "RuntimeClass 'gvisor' missing; apply $INFRA_MANIFEST_DIR/runtimeclass-gvisor.yaml first"
    fi

    # Egress enforcement ships in the agent-egress example layered on top
    # of the solution (auto-detects compute mode and applies the right
    # enforcement layer: Cilium CNPs on Standard EKS, ApplicationNetworkPolicy
    # on Auto Mode). The example must be installed for Step 4/5 of the
    # reference agent to produce BLOCKED outcomes.
    local cilium_policy_ok=""
    local anp_policy_ok=""
    kubectl -n "$NS" get ciliumnetworkpolicy sandbox-llm-allowlist >/dev/null 2>&1 && cilium_policy_ok="yes"
    kubectl -n "$NS" get applicationnetworkpolicy sandbox-llm-allowlist >/dev/null 2>&1 && anp_policy_ok="yes"
    if [ -z "$cilium_policy_ok" ] && [ -z "$anp_policy_ok" ]; then
        fail "No egress allowlist found. Install the egress example first: $AGENT_DIR/egress/install.sh"
    fi
    if [ -n "$cilium_policy_ok" ]; then
        log "Detected Cilium-based egress (Standard EKS — CNP sandbox-llm-allowlist)."
    fi
    if [ -n "$anp_policy_ok" ]; then
        log "Detected native ANP egress (Auto Mode — ApplicationNetworkPolicy sandbox-llm-allowlist)."
    fi
}

setup_configmap_with_real_agent() {
    # sandbox-agent.yaml ships as a SandboxClaim with a templateRef
    # placeholder and a placeholder ConfigMap. We:
    #   1. Render the manifest with the resolved SandboxTemplate name
    #      and apply it (creates SA + placeholder ConfigMap + Claim).
    #   2. Overwrite the ConfigMap with the real agent.py contents.
    #   3. Delete the controller-owned pod so it's recreated and the
    #      container's startup `cp /config/agent.py /workspace/agent.py`
    #      picks up the real content.
    RENDERED_SANDBOX_MANIFEST=$(mktemp -t agent-sandbox-claim.XXXXXX.yaml)
    sed -e "s|__SANDBOX_TEMPLATE__|$SANDBOX_TEMPLATE|g" \
        "$BLUEPRINT_MANIFEST_DIR/sandbox-agent.yaml" \
        > "$RENDERED_SANDBOX_MANIFEST"

    log "Applying SandboxClaim (template: $SANDBOX_TEMPLATE)..."
    kubectl apply -f "$RENDERED_SANDBOX_MANIFEST" >/dev/null

    log "Replacing placeholder ConfigMap with real agent.py contents..."
    kubectl -n "$NS" create configmap "$CONFIGMAP" \
        --from-file=agent.py="$AGENT_DIR/agent.py" \
        --dry-run=client -o yaml | kubectl apply -f - >/dev/null

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
    # regular network path (covered by the egress allowlist) and works
    # transparently on both tiers.
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
    # The controller recreates the pod after our delete; give it a
    # moment to spawn a fresh one before waiting on Ready.
    sleep 5
    if ! kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=300s >/dev/null; then
        kubectl -n "$NS" describe "pod/$POD" >&2
        fail "Sandbox pod did not become Ready within 5 min"
    fi
    log "Pod Ready."
}

assert_runtime_class() {
    # Mode-aware runtime class assertion:
    #   - Standard EKS → expect runtimeClassName=gvisor
    #   - Auto Mode    → expect empty (default runc)
    local rc expected
    rc=$(kubectl -n "$NS" get "pod/$POD" -o jsonpath='{.spec.runtimeClassName}')
    if [ "$COMPUTE_MODE" = "automode" ]; then
        expected=""
        log "Asserting pod runs on default runtime (Auto Mode does not support gVisor)..."
        [ "$rc" = "$expected" ] || fail "Expected empty runtimeClassName on Auto Mode, got '$rc'"
    else
        expected="gvisor"
        log "Asserting pod is scheduled with runtimeClassName=gvisor..."
        [ "$rc" = "$expected" ] || fail "Expected runtimeClassName=gvisor, got '$rc'"
    fi
}

assert_policies_valid() {
    log "Asserting egress policies are Valid..."
    # Policies live in the installed egress example. Check whichever
    # backend is present — the agent-egress example auto-detects mode
    # and applies the matching backend; both share the same resource
    # names (admin-block-imds + sandbox-llm-allowlist).
    local cilium_admin cilium_app anp_admin anp_app
    cilium_admin=$(kubectl get ciliumclusterwidenetworkpolicy admin-block-imds -o jsonpath='{.status.conditions[?(@.type=="Valid")].status}' 2>/dev/null || echo "")
    cilium_app=$(kubectl -n "$NS" get ciliumnetworkpolicy sandbox-llm-allowlist -o jsonpath='{.status.conditions[?(@.type=="Valid")].status}' 2>/dev/null || echo "")
    anp_admin=$(kubectl get clusternetworkpolicy admin-block-imds 2>/dev/null || echo "")
    anp_app=$(kubectl -n "$NS" get applicationnetworkpolicy sandbox-llm-allowlist 2>/dev/null || echo "")

    if [ "$cilium_admin" = "True" ] && [ "$cilium_app" = "True" ]; then
        log "  Cilium-based egress — admin CCNP + app CNP both Valid."
        return 0
    fi
    if [ -n "$anp_admin" ] && [ -n "$anp_app" ]; then
        log "  Native ANP egress — ClusterNetworkPolicy + ApplicationNetworkPolicy both present."
        return 0
    fi
    fail "Expected egress policies not found. For Cilium: admin-block-imds (CCNP) + sandbox-llm-allowlist (CNP). For ANP: admin-block-imds (ClusterNetworkPolicy) + sandbox-llm-allowlist (ApplicationNetworkPolicy)."
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
    # Default: leave the sandbox running so repeat conformance runs are
    # fast (no re-provisioning gVisor nodes). Pass CLEANUP=1 to tear down
    # the sandbox resources on exit. The rendered claim manifest is
    # always cleaned up.
    if [ "${CLEANUP:-0}" = "1" ]; then
        log "Removing test-run resources (SandboxClaim + ConfigMap)..."
        if [ -n "$RENDERED_SANDBOX_MANIFEST" ] && [ -f "$RENDERED_SANDBOX_MANIFEST" ]; then
            kubectl delete -f "$RENDERED_SANDBOX_MANIFEST" --ignore-not-found >/dev/null 2>&1 || true
        fi
        kubectl -n "$NS" delete configmap "$CONFIGMAP" --ignore-not-found >/dev/null 2>&1 || true
        log "Cleanup complete. IAM role + IRSA annotation retained for re-runs."
    else
        log "Leaving Sandbox + ConfigMap in place (set CLEANUP=1 to remove)."
    fi
    if [ -n "$RENDERED_SANDBOX_MANIFEST" ] && [ -f "$RENDERED_SANDBOX_MANIFEST" ]; then
        rm -f "$RENDERED_SANDBOX_MANIFEST"
    fi
}

main() {
    trap cleanup EXIT
    require_env
    detect_compute_mode
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
