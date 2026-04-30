#!/bin/bash
# Agent Sandbox blueprint installer.
#
# Installs an EKS cluster with Karpenter plus the Kubernetes SIG-Apps
# agent-sandbox operator, two sandbox isolation tiers (standard + gVisor),
# and chained Cilium for FQDN egress. Matches the existing blueprint
# pattern: copies the shared base into a local directory, runs its
# install, then layers agent-sandbox-specific manifests on top.
#
# Usage (full install):
#   cd infra/agent-sandbox
#   ./install.sh
#
# Usage (phased — useful when iterating on manifests during demo prep):
#   ./install.sh cluster    # Base EKS cluster only (20-30 min)
#   ./install.sh cilium     # + Cilium chaining + Hubble
#   ./install.sh sandbox    # + SIG-Apps agent-sandbox controller
#   ./install.sh manifests  # + RuntimeClass, SandboxTemplates, NetworkPolicies
#
# Each phase is idempotent; running `./install.sh manifests` after a
# full `./install.sh` just re-applies manifests. Useful when the demo
# needs a last-minute reset.
#
# Destroy:
#   cd infra/agent-sandbox/terraform/_LOCAL
#   ./cleanup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHASE="${1:-all}"

cd "$SCRIPT_DIR"

REGION=$(awk -F'=' '/^region/ {gsub(/[" ]/, "", $2); print $2}' terraform/blueprint.tfvars 2>/dev/null)
REGION=${REGION:-us-east-1}
CLUSTER_NAME=$(awk -F'=' '/^name/ {gsub(/[" ]/, "", $2); print $2}' terraform/blueprint.tfvars 2>/dev/null)
CLUSTER_NAME=${CLUSTER_NAME:-agent-sandbox}

install_cluster() {
    echo "=== Phase 1: Provisioning base EKS cluster ==="
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

install_cilium() {
    echo ""
    echo "=== Phase 2: Installing Cilium (chaining mode) + Hubble ==="
    helm repo add cilium https://helm.cilium.io/ 2>/dev/null || true
    helm repo update cilium
    # Chaining mode: VPC CNI keeps allocating pod IPs + setting up the
    # veth pair, Cilium runs as a meta-plugin attaching eBPF programs
    # on top. Hubble is bundled for flow observability — the primary
    # demo surface for "show me the egress decisions happening in real
    # time."
    #
    # Critical chaining-mode settings that Cilium's Helm defaults get
    # wrong if you don't set them explicitly:
    #   - ipam.mode=cluster-pool (Cilium default) + a pod CIDR that
    #     doesn't collide with the VPC. In aws-cni chaining mode,
    #     this CIDR is NOT used for pod IPs (VPC CNI allocates those)
    #     — it's only for Cilium's per-node `cilium_host` interface
    #     that lives in the host network namespace. Use 10.100.0.0/16
    #     which is outside both our primary VPC CIDR (10.0.0.0/16)
    #     and the secondary (100.64.0.0/16).
    #     The ipam.mode=kubernetes alternative needs every Node to
    #     have spec.podCIDR populated, which EKS does not do.
    #     The ipam.mode=delegated-plugin alternative needs per-node
    #     local-router-ipv4 annotation, which is too intricate for
    #     this blueprint.
    #   - routingMode=native — no encapsulation; VPC CNI handles all
    #     underlay networking. (tunnel=disabled was removed in 1.15;
    #     routingMode=native implies it.)
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

install_sandbox() {
    echo ""
    echo "=== Phase 3: Installing SIG-Apps agent-sandbox (v0.4.3) ==="
    AGENT_SANDBOX_VERSION="v0.4.3"
    kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/manifest.yaml"
    kubectl apply -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${AGENT_SANDBOX_VERSION}/extensions.yaml"
    # Wait for the controller deployments to become ready so subsequent
    # SandboxTemplate applies land against a live webhook.
    kubectl -n agent-sandbox-system wait --for=condition=Available \
        deployment --all --timeout=3m
}

install_manifests() {
    echo ""
    echo "=== Phase 4: Applying runtime class, Karpenter NodePool, sandbox templates, and policies ==="
    kubectl apply -f "$SCRIPT_DIR/manifests/namespace.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/runtimeclass-gvisor.yaml"

    # Karpenter NodePool + EC2NodeClass reference the cluster name
    # explicitly (for subnet/SG tag discovery) AND the exact node
    # IAM role name (the ai-on-eks base module appends a
    # Terraform-generated suffix so the name isn't predictable).
    # Render both from the live cluster state rather than
    # requiring the demo operator to hand-edit.
    #
    # Using sed-based substitution (not envsubst) because the user-data
    # in the NodePool is shell script with its own $VAR references
    # that envsubst would clobber. The __PLACEHOLDER__ token shape is
    # inert to shell.
    export CLUSTER_NAME
    KARPENTER_NODE_ROLE=$(kubectl get ec2nodeclass m6i-cpu -o jsonpath='{.spec.role}' 2>/dev/null)
    if [ -z "$KARPENTER_NODE_ROLE" ]; then
        echo "ERROR: Could not resolve Karpenter node role from EC2NodeClass m6i-cpu."
        echo "       Ensure the base module's Karpenter resources are up first."
        exit 1
    fi
    echo "Karpenter node role: $KARPENTER_NODE_ROLE"
    sed -e "s|__CLUSTER_NAME__|$CLUSTER_NAME|g" \
        -e "s|__KARPENTER_NODE_ROLE__|$KARPENTER_NODE_ROLE|g" \
        "$SCRIPT_DIR/manifests/karpenter-nodepool-gvisor.yaml" | kubectl apply -f -

    kubectl apply -f "$SCRIPT_DIR/manifests/sandbox-template-standard.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/sandbox-template-gvisor.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/ciliumclusterwidenetworkpolicy-admin.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/ciliumnetworkpolicy-sandbox-llm.yaml"

    echo ""
    echo "=== Verifying installation ==="
    kubectl get runtimeclasses
    kubectl get nodepools 2>/dev/null || true
    kubectl get sandboxtemplates -A 2>/dev/null || true
    kubectl get ciliumclusterwidenetworkpolicies 2>/dev/null || true
    kubectl get ciliumnetworkpolicies -A 2>/dev/null || true
}

install_kro() {
    echo ""
    echo "=== Phase 5 (stretch): Installing KRO + AgentSandbox RGD ==="
    bash "$SCRIPT_DIR/manifests/kro-install.sh"
    kubectl apply -f "$SCRIPT_DIR/manifests/rgd-agent-sandbox.yaml"
    # Wait for the RGD's generated CRD to register. Useful to confirm
    # before the showcase since users will immediately kubectl apply
    # the instance YAML.
    echo ""
    echo "Waiting for AgentSandbox CRD to be installed..."
    for i in $(seq 1 30); do
        if kubectl get crd agentsandboxes.custom.agents.x-k8s.io >/dev/null 2>&1; then
            echo "AgentSandbox CRD ready."
            break
        fi
        sleep 2
    done
    kubectl get rgd
}

finish_message() {
    echo ""
    echo "=== Installation complete ==="
    echo ""
    echo "Next steps:"
    echo "  - Run the reference agent:    kubectl apply -f $SCRIPT_DIR/manifests/demo-agent.yaml"
    echo "  - Watch sandbox provisioning: kubectl get sandboxes -A -w"
    echo "  - Open Hubble UI:             kubectl port-forward -n kube-system svc/hubble-ui 12000:80"
    echo "  - Cleanup:                    cd terraform/_LOCAL && ./cleanup.sh"
}

case "$PHASE" in
    cluster)
        install_cluster
        ;;
    cilium)
        install_cilium
        ;;
    sandbox)
        install_sandbox
        ;;
    manifests)
        install_manifests
        finish_message
        ;;
    kro)
        install_kro
        ;;
    all)
        install_cluster
        install_cilium
        install_sandbox
        install_manifests
        finish_message
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: cluster | cilium | sandbox | manifests | kro | all"
        exit 1
        ;;
esac
