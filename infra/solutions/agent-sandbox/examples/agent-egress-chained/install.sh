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
#   ./install.sh                # Install Cilium + apply policies
#   ./install.sh cilium         # Cilium chaining + Hubble only
#   ./install.sh policies       # Policies only (Cilium already installed)
#   ./install.sh uninstall      # Remove policies + Cilium

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-install}"

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

uninstall() {
    echo "=== Removing CNPs and Cilium ==="
    kubectl delete -f "$SCRIPT_DIR/manifests/ciliumnetworkpolicy-sandbox-llm.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/manifests/ciliumclusterwidenetworkpolicy-admin.yaml" --ignore-not-found
    helm uninstall cilium -n kube-system || true
    echo "Uninstall complete."
}

finish_message() {
    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "Next steps:"
    echo "  - Open the Hubble UI:          kubectl port-forward -n kube-system svc/hubble-ui 12000:80"
    echo "  - Browse allowlist templates:  ls manifests/allowlists/"
    echo "  - Run end-to-end conformance:  cd ../../../../../blueprints/agents/agent-sandbox && ./conformance.sh"
}

case "$PHASE" in
    cilium)
        install_cilium
        ;;
    policies)
        install_policies
        ;;
    install)
        install_cilium
        install_policies
        restart_sandbox_controller
        finish_message
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: cilium | policies | install | uninstall"
        exit 1
        ;;
esac
