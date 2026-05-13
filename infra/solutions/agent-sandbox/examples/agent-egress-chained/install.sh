#!/bin/bash
# Agent Sandbox — Chained egress example.
#
# Installs Cilium in aws-cni chaining mode on top of the base EKS VPC
# CNI, plus Hubble for flow observability, plus the admin-tier
# CiliumClusterwideNetworkPolicy + sandbox-tier CiliumNetworkPolicy
# for FQDN egress enforcement. Use this example on Standard EKS
# clusters (not Auto Mode) — native ApplicationNetworkPolicy is
# available only on EKS Auto Mode and is the canonical path there;
# this example is the canonical path for Standard EKS.
#
# Cilium is one of several service-mesh options that can provide FQDN
# filtering — Istio, Linkerd, and others support similar patterns via
# chaining. Cilium is used here for convenience (single dependency,
# native Hubble observability, stable CNCF-graduated project), not
# out of architectural necessity.
#
# When you migrate to EKS Auto Mode (where ApplicationNetworkPolicy
# is available), you can switch from this example to the sibling
# agent-egress-native by replacing the CiliumNetworkPolicy manifests
# with equivalent
# ApplicationNetworkPolicy resources. The allowlist templates ship
# as CNP/ANP pairs so migration is mechanical.
#
# Prerequisite: the parent agent-sandbox solution must be deployed
# first (provides the cluster + agent-sandboxes namespace + SIG-Apps
# controller). This example assumes kubectl is configured for that
# cluster.
#
# Usage:
#   cd infra/solutions/agent-sandbox/examples/agent-egress-chained
#   ./install.sh                # Cilium + policies + IRSA + controller bounce
#   ./install.sh cilium         # Cilium chaining + Hubble only
#   ./install.sh policies       # Policies only (Cilium already installed)
#   ./install.sh irsa           # Bedrock IRSA role only (idempotent)
#   ./install.sh uninstall      # Remove policies + Cilium + IRSA role

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

install_cilium() {
    echo "=== Installing Cilium (chaining mode) + Hubble ==="
    helm repo add cilium https://helm.cilium.io/ 2>/dev/null || true
    helm repo update cilium
    # Chaining mode: VPC CNI keeps allocating pod IPs + setting up the
    # veth pair, Cilium runs as a meta-plugin attaching eBPF programs
    # on top. Hubble is bundled for flow observability.
    #
    # Critical chaining-mode settings that Cilium's Helm defaults get
    # wrong if you don't set them explicitly:
    #   - ipam.mode=cluster-pool (Cilium default) + a pod CIDR that
    #     doesn't collide with the VPC. In aws-cni chaining mode,
    #     this CIDR is NOT used for pod IPs (VPC CNI allocates those)
    #     — it's only for Cilium's per-node `cilium_host` interface
    #     that lives in the host network namespace. Use 10.100.0.0/16
    #     which is outside both common VPC CIDRs (10.0.0.0/16 primary,
    #     100.64.0.0/16 EKS secondary).
    #   - routingMode=native — no encapsulation; VPC CNI handles all
    #     underlay networking.
    #   - kubeProxyReplacement=false — chaining doesn't replace
    #     kube-proxy; VPC CNI relies on it.
    #   - enableIPv4Masquerade=false — VPC CNI handles SNAT via the
    #     underlying ENI.
    helm upgrade --install cilium cilium/cilium \
        --namespace kube-system \
        --set cni.chainingMode=aws-cni \
        --set cni.exclusive=false \
        --set ipam.operator.clusterPoolIPv4PodCIDRList="{10.100.0.0/16}" \
        --set endpointRoutes.enabled=true \
        --set routingMode=native \
        --set kubeProxyReplacement=false \
        --set enableIPv4Masquerade=false \
        --set l7Proxy=true \
        --set hubble.enabled=true \
        --set hubble.relay.enabled=true \
        --set hubble.ui.enabled=true \
        --set hubble.metrics.enabled="{dns,drop,tcp,flow,port-distribution,icmp,http}" \
        --wait --timeout 5m
    kubectl -n kube-system rollout status daemonset cilium --timeout=2m
}

install_policies() {
    echo ""
    echo "=== Applying admin + app-tier CiliumNetworkPolicies ==="
    kubectl apply -f "$SCRIPT_DIR/manifests/ciliumclusterwidenetworkpolicy-admin.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/ciliumnetworkpolicy-sandbox-llm.yaml"

    echo ""
    echo "=== Verifying installation ==="
    kubectl get ciliumclusterwidenetworkpolicies 2>/dev/null || true
    kubectl get ciliumnetworkpolicies -A 2>/dev/null || true
}

restart_sandbox_controller() {
    # Bounce the agent-sandbox controller so it reconnects through
    # the new chained datapath. Cilium's chaining install replaces
    # the eBPF programs on every veth, and any pod that opened its
    # kube-API connection before chaining (including the
    # agent-sandbox controller deployed by the parent solution's
    # ArgoCD addon) holds a stale connection that won't recover on
    # its own — symptom is the controller logging "context deadline
    # exceeded" against kube-apiserver and failing to reconcile new
    # Sandbox resources.
    #
    # Skipped silently if the deployment isn't present (parent
    # solution may have been deployed without enable_agent_sandbox).
    if kubectl -n agent-sandbox-system get deployment agent-sandbox-controller >/dev/null 2>&1; then
        echo ""
        echo "=== Bouncing agent-sandbox controller (post-Cilium chaining) ==="
        kubectl -n agent-sandbox-system rollout restart deployment agent-sandbox-controller
        kubectl -n agent-sandbox-system rollout status deployment agent-sandbox-controller --timeout=2m
    fi
}

# Idempotent provisioning of the Bedrock IRSA role used by the
# reference agent. Resolves cluster + region + account + OIDC
# provider from the live state, renders the trust + permissions
# templates, then either creates the role (first run) or updates
# the trust policy (re-runs after cluster recreation, where the
# OIDC provider ID has changed). Always re-attaches the inline
# permission policy so it stays in sync with the template.
#
# The function is deliberately copy-shared with the native
# example's install.sh — both examples need this, neither is the
# canonical place to centralize it (since users only run one or
# the other), and a sourced helper would introduce a third file.
# If divergence becomes desirable it should be intentional, not
# accidental drift.
bootstrap_irsa() {
    resolve_cluster_context
    local trust_template="$SOLUTION_DIR/manifests/iam-bedrock-trust-policy.template.json"
    local perms_template="$SOLUTION_DIR/manifests/iam-bedrock-permissions.template.json"
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
    echo "=== Removing CNPs and Cilium ==="
    kubectl delete -f "$SCRIPT_DIR/manifests/ciliumnetworkpolicy-sandbox-llm.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/manifests/ciliumclusterwidenetworkpolicy-admin.yaml" --ignore-not-found
    helm uninstall cilium -n kube-system || true
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
    echo "Next steps:"
    echo "  - Open the Hubble UI:          kubectl port-forward -n kube-system svc/hubble-ui 12000:80"
    echo "  - Browse allowlist templates:  ls manifests/allowlists/"
    echo "  - Run end-to-end conformance:  cd ../../../../../blueprints/agents/agent-sandbox && \\"
    echo "      BEDROCK_ROLE_ARN=$BEDROCK_ROLE_ARN ./conformance.sh"
}

case "$PHASE" in
    cilium)
        install_cilium
        ;;
    policies)
        install_policies
        ;;
    irsa)
        bootstrap_irsa
        ;;
    install)
        install_cilium
        install_policies
        restart_sandbox_controller
        bootstrap_irsa
        finish_message
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: cilium | policies | irsa | install | uninstall"
        exit 1
        ;;
esac
