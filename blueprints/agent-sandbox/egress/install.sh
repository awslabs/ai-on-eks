#!/bin/bash
# Agent Sandbox — Egress enforcement example.
#
# Auto-detects compute mode at apply time and applies the right
# enforcement layer:
#   - Standard EKS → CiliumNetworkPolicy (Cilium installed by base
#     infra when enable_cilium=true; this script applies CNPs +
#     bounces the agent-sandbox controller post-chaining).
#   - EKS Auto Mode → ApplicationNetworkPolicy (DNS-based filter
#     enforced by VPC CNI Network Policy Controller; this script
#     enables the controller + applies ANPs).
#
# Both paths share identical pod-level allowlist labels
# (`allowlist: <name>`), so agent workloads are portable between
# the two backends without relabeling.
#
# Prerequisite: the parent agent-sandbox solution must be deployed
# first. For Standard EKS, also requires `enable_cilium = true` in
# the solution's blueprint.tfvars (the base infra ArgoCD-deploys
# Cilium chaining mode + Hubble). For Auto Mode, set
# `enable_eks_auto_mode = true` and `enable_cilium = false` (no
# third-party CNI required).
#
# Usage:
#   cd blueprints/agent-sandbox/egress
#   ./install.sh                # Mode detection + policies + IRSA
#   ./install.sh policies       # Policies only (mode-aware)
#   ./install.sh irsa           # Bedrock IRSA role only (idempotent)
#   ./install.sh uninstall      # Remove policies + IRSA role

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Egress lives at blueprints/agent-sandbox/egress/. The blueprint root
# (containing the IAM templates) is one level up.
BLUEPRINT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PHASE="${1:-install}"

# Resolved on demand by phases that need it. Single source of truth so
# phases agree on the cluster context even if kubectl context changes
# between calls.
resolve_cluster_context() {
    CLUSTER_NAME="${CLUSTER_NAME:-$(kubectl config current-context | awk -F'/' '{print $NF}')}"
    REGION="${REGION:-$(kubectl config current-context | awk -F':' '{print $4}')}"
    ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
    BEDROCK_ROLE_NAME="${BEDROCK_ROLE_NAME:-${CLUSTER_NAME}-bedrock-irsa}"
}

# Detect compute mode at apply time. Standard EKS (default) → cilium
# enforcement path; EKS Auto Mode → ANP enforcement path. Retries
# transient AWS API failures up to 3 times — false-negatives that
# route to the wrong mode are worse than slow detection.
detect_compute_mode() {
    resolve_cluster_context
    local auto_mode attempt
    for attempt in 1 2 3; do
        auto_mode=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
            --query 'cluster.computeConfig.enabled' --output text 2>/dev/null || echo "")
        if [ "$auto_mode" = "True" ] || [ "$auto_mode" = "true" ]; then
            COMPUTE_MODE="auto"
            return 0
        fi
        if [ "$auto_mode" = "False" ] || [ "$auto_mode" = "false" ]; then
            COMPUTE_MODE="standard"
            return 0
        fi
        # Empty or unexpected response — could be transient API failure.
        if [ "$attempt" -lt 3 ]; then
            sleep 2
        fi
    done
    echo "ERROR: Could not determine cluster compute mode after 3 retries."
    echo "       Cluster: $CLUSTER_NAME ($REGION)"
    echo "       Verify AWS credentials and EKS describe-cluster permissions."
    exit 1
}

install_policies_standard() {
    # Standard EKS path — assumes Cilium has been deployed by the base
    # infra (enable_cilium=true in blueprint.tfvars).
    echo "=== Standard EKS detected — Cilium enforcement path ==="
    if ! kubectl -n kube-system get deployment cilium-operator >/dev/null 2>&1; then
        echo ""
        echo "ERROR: Cilium operator not found in kube-system."
        echo "       Set enable_cilium=true in the parent solution's blueprint.tfvars"
        echo "       and re-run its install.sh, then re-run this script."
        exit 1
    fi

    echo ""
    echo "=== Applying admin + app-tier CiliumNetworkPolicies ==="
    kubectl apply -f "$SCRIPT_DIR/manifests/cilium/ciliumclusterwidenetworkpolicy-admin.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/cilium/ciliumnetworkpolicy-sandbox-llm.yaml"

    # Bounce the agent-sandbox controller so it reconnects through
    # the chained datapath. Cilium's chaining install replaces the
    # eBPF programs on every veth, and any pod that opened its
    # kube-API connection before chaining holds a stale connection
    # that won't recover on its own.
    if kubectl -n agent-sandbox-system get deployment agent-sandbox-controller >/dev/null 2>&1; then
        echo ""
        echo "=== Bouncing agent-sandbox controller (post-Cilium chaining) ==="
        kubectl -n agent-sandbox-system rollout restart deployment agent-sandbox-controller
        kubectl -n agent-sandbox-system rollout status deployment agent-sandbox-controller --timeout=2m
    fi

    echo ""
    echo "=== Verifying installation ==="
    kubectl get ciliumclusterwidenetworkpolicies 2>/dev/null || true
    kubectl get ciliumnetworkpolicies -A 2>/dev/null || true
}

install_policies_auto() {
    # Auto Mode path — enables Network Policy Controller + applies ANPs.
    echo "=== EKS Auto Mode detected — ApplicationNetworkPolicy enforcement path ==="
    echo ""
    echo "=== Enabling Network Policy Controller ==="
    # Required for ApplicationNetworkPolicy / ClusterNetworkPolicy
    # enforcement on Auto Mode. CRDs are pre-installed but the
    # controller is disabled by default; applying this ConfigMap
    # activates enforcement.
    kubectl apply -f "$SCRIPT_DIR/manifests/anp/network-policy-controller-enable.yaml"

    echo ""
    echo "=== Applying admin-tier ClusterNetworkPolicy + app-tier ApplicationNetworkPolicy ==="
    kubectl apply -f "$SCRIPT_DIR/manifests/anp/clusternetworkpolicy-admin.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/anp/applicationnetworkpolicy-sandbox-llm.yaml"

    echo ""
    echo "=== Verifying installation ==="
    kubectl get clusternetworkpolicies 2>/dev/null || true
    kubectl get applicationnetworkpolicies -A 2>/dev/null || true
}

install_policies() {
    detect_compute_mode
    if [ "$COMPUTE_MODE" = "auto" ]; then
        install_policies_auto
    else
        install_policies_standard
    fi
}

# Idempotent provisioning of the Bedrock IRSA role used by the
# reference agent. Resolves cluster + region + account + OIDC
# provider from the live state, renders the trust + permissions
# templates, then either creates the role (first run) or updates
# the trust policy (re-runs after cluster recreation, where the
# OIDC provider ID has changed). Always re-attaches the inline
# permission policy so it stays in sync with the template.
bootstrap_irsa() {
    resolve_cluster_context
    local trust_template="$BLUEPRINT_DIR/manifests/iam/bedrock-trust-policy.template.json"
    local perms_template="$BLUEPRINT_DIR/manifests/iam/bedrock-permissions.template.json"
    local trust_rendered=$(mktemp -t agent-sandbox-trust.XXXXXX.json)
    local perms_rendered=$(mktemp -t agent-sandbox-perms.XXXXXX.json)
    trap "rm -f $trust_rendered $perms_rendered" RETURN

    local oidc_issuer oidc_provider_id
    oidc_issuer=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
        --query 'cluster.identity.oidc.issuer' --output text)
    oidc_provider_id=$(echo "$oidc_issuer" | awk -F'/' '{print $NF}')

    echo "=== Bedrock IRSA role ==="
    echo "  Cluster:     $CLUSTER_NAME ($REGION)"
    echo "  Account:     $ACCOUNT_ID"
    echo "  OIDC ID:     $oidc_provider_id"
    echo "  Role name:   $BEDROCK_ROLE_NAME"

    # Render templates. The IAM API rejects the `Comment` field that
    # the templates carry as self-documentation, so strip it via jq.
    jq 'del(.Comment)' "$trust_template" \
        | sed -e "s|<ACCOUNT_ID>|$ACCOUNT_ID|g" \
              -e "s|<REGION>|$REGION|g" \
              -e "s|<OIDC_PROVIDER_ID>|$oidc_provider_id|g" \
        > "$trust_rendered"
    jq 'del(.Comment)' "$perms_template" \
        | sed -e "s|<ACCOUNT_ID>|$ACCOUNT_ID|g" \
        > "$perms_rendered"

    if aws iam get-role --role-name "$BEDROCK_ROLE_NAME" >/dev/null 2>&1; then
        echo "  Role exists — updating trust policy (handles OIDC drift after cluster recreation)..."
        aws iam update-assume-role-policy --role-name "$BEDROCK_ROLE_NAME" \
            --policy-document "file://$trust_rendered" >/dev/null
    else
        echo "  Creating role..."
        aws iam create-role --role-name "$BEDROCK_ROLE_NAME" \
            --assume-role-policy-document "file://$trust_rendered" \
            --description "IRSA role for agent-sandbox reference agent - Bedrock invoke" \
            --query 'Role.Arn' --output text >/dev/null
    fi

    echo "  Attaching BedrockInvoke inline policy..."
    aws iam put-role-policy --role-name "$BEDROCK_ROLE_NAME" \
        --policy-name BedrockInvoke \
        --policy-document "file://$perms_rendered"

    BEDROCK_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${BEDROCK_ROLE_NAME}"
    echo "  Role ARN:    $BEDROCK_ROLE_ARN"
    echo ""
    echo "  Export this for conformance.sh:"
    echo "    export BEDROCK_ROLE_ARN=$BEDROCK_ROLE_ARN"
}

uninstall_irsa() {
    resolve_cluster_context
    if aws iam get-role --role-name "$BEDROCK_ROLE_NAME" >/dev/null 2>&1; then
        echo "  Deleting BedrockInvoke inline policy..."
        aws iam delete-role-policy --role-name "$BEDROCK_ROLE_NAME" \
            --policy-name BedrockInvoke 2>/dev/null || true
        echo "  Deleting role $BEDROCK_ROLE_NAME..."
        aws iam delete-role --role-name "$BEDROCK_ROLE_NAME"
    else
        echo "  Role $BEDROCK_ROLE_NAME does not exist — skipping."
    fi
}

uninstall() {
    detect_compute_mode
    echo "=== Removing egress policies ($COMPUTE_MODE mode) ==="
    if [ "$COMPUTE_MODE" = "auto" ]; then
        kubectl delete -f "$SCRIPT_DIR/manifests/anp/applicationnetworkpolicy-sandbox-llm.yaml" --ignore-not-found
        kubectl delete -f "$SCRIPT_DIR/manifests/anp/clusternetworkpolicy-admin.yaml" --ignore-not-found
        echo "  Network Policy Controller ConfigMap left in place (other workloads may depend on it)."
    else
        kubectl delete -f "$SCRIPT_DIR/manifests/cilium/ciliumnetworkpolicy-sandbox-llm.yaml" --ignore-not-found
        kubectl delete -f "$SCRIPT_DIR/manifests/cilium/ciliumclusterwidenetworkpolicy-admin.yaml" --ignore-not-found
        echo "  Cilium left in place (deployed via base infra; disable via enable_cilium=false in blueprint.tfvars)."
    fi
    echo ""
    echo "=== Removing Bedrock IRSA role ==="
    uninstall_irsa
    echo ""
    echo "Uninstall complete."
}

finish_message() {
    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "Compute mode: $COMPUTE_MODE"
    echo ""
    echo "Next steps:"
    if [ "$COMPUTE_MODE" = "standard" ]; then
        echo "  - Open Hubble UI:              kubectl port-forward -n kube-system svc/hubble-ui 12000:80"
    fi
    echo "  - Browse allowlist templates:  ls manifests/allowlists/$COMPUTE_MODE_DIR/"
    echo "  - Run end-to-end conformance:  cd .. && \\"
    echo "      BEDROCK_ROLE_ARN=$BEDROCK_ROLE_ARN ./conformance.sh"
}

case "$PHASE" in
    install)
        install_policies
        bootstrap_irsa
        # Map COMPUTE_MODE to the manifests subdirectory name for the
        # finish_message hint.
        if [ "$COMPUTE_MODE" = "auto" ]; then
            COMPUTE_MODE_DIR="anp"
        else
            COMPUTE_MODE_DIR="cilium"
        fi
        finish_message
        ;;
    policies)
        install_policies
        ;;
    irsa)
        bootstrap_irsa
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: install | policies | irsa | uninstall"
        exit 1
        ;;
esac
