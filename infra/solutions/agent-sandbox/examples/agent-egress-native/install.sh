#!/bin/bash
# Agent Sandbox — Native egress example.
#
# Applies EKS-native egress enforcement (ClusterNetworkPolicy +
# ApplicationNetworkPolicy) on top of an Auto Mode cluster deployed
# by the parent solution.
#
# Requires EKS Auto Mode — DNS-based ANP rules enforce only on Auto
# Mode-launched EC2 instances (per AWS docs as of this blueprint's
# publication). To use this example, flip `enable_eks_auto_mode =
# true` in infra/solutions/agent-sandbox/terraform/blueprint.tfvars
# before running the parent solution's install.sh.
#
# For Standard EKS, use the sibling agent-egress-chained example
# instead (Cilium + Hubble chaining is the canonical FQDN-filtering
# path for Standard EKS clusters).
#
# Usage:
#   cd infra/solutions/agent-sandbox/examples/agent-egress-native
#   ./install.sh                # Apply policies + provision IRSA role
#   ./install.sh policies       # Policies only
#   ./install.sh irsa           # Bedrock IRSA role only (idempotent)
#   ./install.sh uninstall      # Remove policies + IRSA role

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOLUTION_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PHASE="${1:-install}"

# Resolved on demand by phases that need it. Single source of truth so
# the irsa + uninstall phases agree on the role name even if the
# kubectl context changes between calls.
resolve_cluster_context() {
    CLUSTER_NAME="${CLUSTER_NAME:-$(kubectl config current-context | awk -F'/' '{print $NF}')}"
    REGION="${REGION:-$(kubectl config current-context | awk -F':' '{print $4}')}"
    ACCOUNT_ID="${ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
    BEDROCK_ROLE_NAME="${BEDROCK_ROLE_NAME:-${CLUSTER_NAME}-bedrock-irsa}"
}

require_auto_mode() {
    # Confirm the cluster is Auto Mode. ANP enforcement only works on
    # Auto Mode-launched EC2 instances; applying the manifests to
    # Standard EKS creates the CRDs but they won't enforce anything.
    #
    # Retry the describe-cluster call up to 3 times with a brief
    # backoff — transient AWS API failures (DNS hiccups, throttling)
    # would otherwise produce a false-negative and mislead users into
    # thinking they're not on Auto Mode.
    local cluster_name region auto_mode attempt
    cluster_name=$(kubectl config current-context | awk -F'/' '{print $NF}')
    region=$(kubectl config current-context | awk -F':' '{print $4}')
    for attempt in 1 2 3; do
        auto_mode=$(aws eks describe-cluster --name "$cluster_name" --region "$region" \
            --query 'cluster.computeConfig.enabled' --output text 2>/dev/null || echo "")
        if [ "$auto_mode" = "True" ] || [ "$auto_mode" = "true" ]; then
            return 0
        fi
        if [ "$auto_mode" = "False" ] || [ "$auto_mode" = "false" ]; then
            # API explicitly says Standard EKS — don't retry.
            break
        fi
        # Empty or unexpected response — could be transient API failure.
        # Brief backoff and try again.
        if [ "$attempt" -lt 3 ]; then
            sleep 2
        fi
    done
    echo "WARNING: Cluster '$cluster_name' is not in EKS Auto Mode."
    echo "         ApplicationNetworkPolicy DNS-based rules will not enforce."
    echo "         Flip enable_eks_auto_mode=true in the parent solution's"
    echo "         blueprint.tfvars and re-run its install.sh, or use the"
    echo "         sibling agent-egress-chained example for Standard EKS."
    exit 1
}

install_policies() {
    require_auto_mode
    echo "=== Enabling Network Policy Controller ==="
    # Required for ApplicationNetworkPolicy / ClusterNetworkPolicy
    # enforcement on Auto Mode. CRDs are pre-installed but the
    # controller is disabled by default; applying this ConfigMap
    # activates enforcement. See the header comment in
    # manifests/network-policy-controller-enable.yaml.
    kubectl apply -f "$SCRIPT_DIR/manifests/network-policy-controller-enable.yaml"

    echo ""
    echo "=== Applying admin-tier ClusterNetworkPolicy + app-tier ApplicationNetworkPolicy ==="
    kubectl apply -f "$SCRIPT_DIR/manifests/clusternetworkpolicy-admin.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/applicationnetworkpolicy-sandbox-llm.yaml"

    echo ""
    echo "=== Verifying installation ==="
    kubectl get clusternetworkpolicies 2>/dev/null || true
    kubectl get applicationnetworkpolicies -A 2>/dev/null || true
}

# Idempotent provisioning of the Bedrock IRSA role used by the
# reference agent. Resolves cluster + region + account + OIDC
# provider from the live state, renders the trust + permissions
# templates, then either creates the role (first run) or updates
# the trust policy (re-runs after cluster recreation, where the
# OIDC provider ID has changed). Always re-attaches the inline
# permission policy so it stays in sync with the template.
#
# The function is deliberately copy-shared with the chained
# example's install.sh — both examples need this, neither is the
# canonical place to centralize it (since users only run one or
# the other), and a sourced helper would introduce a third file.
# If divergence becomes desirable it should be intentional, not
# accidental drift.
bootstrap_irsa() {
    resolve_cluster_context
    local trust_template="$SOLUTION_DIR/manifests/iam/bedrock-trust-policy.template.json"
    local perms_template="$SOLUTION_DIR/manifests/iam/bedrock-permissions.template.json"
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

finish_message() {
    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "Next steps:"
    echo "  - Browse allowlist templates: ls manifests/allowlists/"
    echo "  - Run end-to-end conformance: cd ../../../../../blueprints/agents/agent-sandbox && \\"
    echo "      BEDROCK_ROLE_ARN=$BEDROCK_ROLE_ARN ./conformance.sh"
}

uninstall() {
    echo "=== Removing ClusterNetworkPolicy + ApplicationNetworkPolicy ==="
    kubectl delete -f "$SCRIPT_DIR/manifests/applicationnetworkpolicy-sandbox-llm.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/manifests/clusternetworkpolicy-admin.yaml" --ignore-not-found
    echo ""
    echo "=== Removing Bedrock IRSA role ==="
    uninstall_irsa
    echo ""
    echo "Uninstall complete. Network Policy Controller ConfigMap left in place (other workloads may depend on it)."
}

case "$PHASE" in
    install)
        install_policies
        bootstrap_irsa
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
