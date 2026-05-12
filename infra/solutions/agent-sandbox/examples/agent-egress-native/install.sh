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
# instead (Cilium + Hubble chaining provides equivalent FQDN
# filtering until AWS extends ANP to Standard EKS).
#
# Usage:
#   cd infra/solutions/agent-sandbox/examples/agent-egress-native
#   ./install.sh                # Apply policies
#   ./install.sh uninstall      # Remove policies

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-install}"

require_auto_mode() {
    # Confirm the cluster is Auto Mode. ANP enforcement only works on
    # Auto Mode-launched EC2 instances; applying the manifests to
    # Standard EKS creates the CRDs but they won't enforce anything.
    local cluster_name region auto_mode
    cluster_name=$(kubectl config current-context | awk -F'/' '{print $NF}')
    region=$(kubectl config current-context | awk -F':' '{print $4}')
    auto_mode=$(aws eks describe-cluster --name "$cluster_name" --region "$region" \
        --query 'cluster.computeConfig.enabled' --output text 2>/dev/null || echo "")
    if [ "$auto_mode" != "True" ] && [ "$auto_mode" != "true" ]; then
        echo "WARNING: Cluster '$cluster_name' is not in EKS Auto Mode."
        echo "         ApplicationNetworkPolicy DNS-based rules will not enforce."
        echo "         Flip enable_eks_auto_mode=true in the parent solution's"
        echo "         blueprint.tfvars and re-run its install.sh, or use the"
        echo "         sibling agent-egress-chained example for Standard EKS."
        exit 1
    fi
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

    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "Next steps:"
    echo "  - Browse allowlist templates: ls manifests/allowlists/"
    echo "  - Run end-to-end conformance: cd ../../../../../blueprints/agents/agent-sandbox && ./conformance.sh"
}

uninstall() {
    echo "=== Removing ClusterNetworkPolicy + ApplicationNetworkPolicy ==="
    kubectl delete -f "$SCRIPT_DIR/manifests/applicationnetworkpolicy-sandbox-llm.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/manifests/clusternetworkpolicy-admin.yaml" --ignore-not-found
    echo "Uninstall complete. Network Policy Controller ConfigMap left in place (other workloads may depend on it)."
}

case "$PHASE" in
    install)
        install_policies
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: install | uninstall"
        exit 1
        ;;
esac
