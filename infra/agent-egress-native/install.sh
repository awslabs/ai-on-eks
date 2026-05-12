#!/bin/bash
# Agent Egress (Native) blueprint installer.
#
# Applies native EKS ClusterNetworkPolicy + ApplicationNetworkPolicy
# for FQDN-aware egress enforcement. Requires EKS Auto Mode —
# DNS-based ANP rules are available only on Auto Mode-launched EC2
# instances (per AWS docs, as of this blueprint's publication).
#
# Target audience: customers running EKS Auto Mode who want FQDN
# egress enforcement without a third-party CNI. Uses the networking
# capability AWS ships natively in VPC CNI v1.21.1+, no Cilium or
# other service-mesh dependency.
#
# Positioning vs the chained blueprint:
#   infra/agent-egress-chained/  — Cilium + Hubble. Standard EKS today.
#   infra/agent-egress-native/   — This blueprint. EKS Auto Mode today.
# When ApplicationNetworkPolicy extends to Standard EKS, customers
# can migrate from chained to native by replacing CNPs with the ANP
# templates shipped here. Pod-level allowlist labels are identical.
#
# Prerequisite: install `infra/agent-sandbox/` first (with
# `enable_eks_auto_mode = true` in its terraform/blueprint.tfvars).
# Alternatively, provision a cluster with THIS blueprint's
# terraform/blueprint.tfvars and then layer agent-sandbox on top.
#
# Usage (full install — applies policies to existing cluster):
#   cd infra/agent-egress-native
#   ./install.sh
#
# Usage (provision Auto Mode cluster + apply policies):
#   ./install.sh cluster   # Provision EKS Auto Mode cluster (20-30 min)
#   ./install.sh policies  # Apply CNP + ANP
#
# Destroy:
#   ./install.sh uninstall      # Remove policies only
#   cd terraform/_LOCAL && ./cleanup.sh    # Destroy cluster

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-policies}"

REGION=$(awk -F'=' '/^region/ {gsub(/[" ]/, "", $2); print $2}' terraform/blueprint.tfvars 2>/dev/null)
REGION=${REGION:-us-east-1}
CLUSTER_NAME=$(awk -F'=' '/^name/ {gsub(/[" ]/, "", $2); print $2}' terraform/blueprint.tfvars 2>/dev/null)
CLUSTER_NAME=${CLUSTER_NAME:-agent-egress-native}

install_cluster() {
    echo "=== Provisioning EKS Auto Mode cluster ==="
    mkdir -p ./terraform/_LOCAL
    cp -r ../base/terraform/* ./terraform/_LOCAL/
    cp "$SCRIPT_DIR/terraform/blueprint.tfvars" ./terraform/_LOCAL/blueprint.tfvars

    cd "$SCRIPT_DIR/terraform/_LOCAL"
    bash ./install.sh

    cd "$SCRIPT_DIR"
    echo ""
    echo "=== Configuring kubectl ==="
    aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME"
    kubectl get nodes
}

require_auto_mode() {
    # Confirm the cluster is Auto Mode. ANP enforcement only works on
    # Auto Mode-launched EC2 instances; applying the manifests to
    # Standard EKS creates the CRDs but they won't enforce anything.
    local auto_mode
    auto_mode=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
        --query 'cluster.computeConfig.enabled' --output text 2>/dev/null || echo "")
    if [ "$auto_mode" != "True" ] && [ "$auto_mode" != "true" ]; then
        echo "WARNING: Cluster '$CLUSTER_NAME' is not in EKS Auto Mode."
        echo "         ApplicationNetworkPolicy DNS-based rules will not enforce."
        echo "         Use infra/agent-egress-chained/ instead, or provision a new"
        echo "         Auto Mode cluster via './install.sh cluster'."
        exit 1
    fi
}

install_policies() {
    require_auto_mode
    echo ""
    echo "=== Applying admin-tier ClusterNetworkPolicy + app-tier ApplicationNetworkPolicy ==="
    kubectl apply -f "$SCRIPT_DIR/manifests/clusternetworkpolicy-admin.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/applicationnetworkpolicy-sandbox-llm.yaml"

    echo ""
    echo "=== Verifying installation ==="
    kubectl get clusternetworkpolicies 2>/dev/null || true
    kubectl get applicationnetworkpolicies -A 2>/dev/null || true
}

uninstall() {
    echo "=== Removing ClusterNetworkPolicy + ApplicationNetworkPolicy ==="
    kubectl delete -f "$SCRIPT_DIR/manifests/applicationnetworkpolicy-sandbox-llm.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/manifests/clusternetworkpolicy-admin.yaml" --ignore-not-found
    echo "Uninstall complete. IAM role + IRSA annotation retained."
}

finish_message() {
    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "Next steps:"
    echo "  - Browse allowlist templates: ls manifests/allowlists/"
    echo "  - Run end-to-end conformance: cd ../agent-sandbox && ./conformance.sh"
}

case "$PHASE" in
    cluster)
        install_cluster
        ;;
    policies)
        install_policies
        finish_message
        ;;
    all)
        install_cluster
        install_policies
        finish_message
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: cluster | policies | all | uninstall"
        exit 1
        ;;
esac
