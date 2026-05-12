#!/bin/bash
# Agent Sandbox blueprint installer.
#
# Installs an EKS cluster with Karpenter plus the Kubernetes SIG-Apps
# agent-sandbox operator and two sandbox isolation tiers (standard +
# gVisor). Matches the existing blueprint pattern: copies the shared
# base into a local directory, runs its install, then layers
# agent-sandbox-specific manifests on top.
#
# This blueprint covers the sandbox runtime only — egress enforcement
# is a separate concern addressed by two sibling blueprints:
#   infra/agent-egress-chained/  — VPC CNI + Cilium chaining, available
#                                  on Standard EKS today.
#   infra/agent-egress-native/   — ApplicationNetworkPolicy via VPC CNI,
#                                  available on EKS Auto Mode today.
# Install agent-sandbox first, then install one of the egress
# blueprints against the same cluster.
#
# Usage (full install):
#   cd infra/agent-sandbox
#   ./install.sh
#
# Usage (phased — useful when iterating on manifests):
#   ./install.sh cluster    # Base EKS cluster only (20-30 min)
#   ./install.sh sandbox    # + SIG-Apps agent-sandbox controller
#   ./install.sh manifests  # + RuntimeClass, NodePool, SandboxTemplates
#   ./install.sh kro        # + KRO + AgentSandbox RGD (optional)
#
# Each phase is idempotent; running `./install.sh manifests` after a
# full `./install.sh` just re-applies manifests. Useful when
# debugging or reprovisioning individual components.
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

install_sandbox() {
    echo ""
    echo "=== Phase 2: Installing SIG-Apps agent-sandbox (v0.4.3) ==="
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
    echo "=== Phase 3: Applying runtime class, Karpenter NodePool, and sandbox templates ==="
    kubectl apply -f "$SCRIPT_DIR/manifests/namespace.yaml"
    kubectl apply -f "$SCRIPT_DIR/manifests/runtimeclass-gvisor.yaml"

    # Karpenter NodePool + EC2NodeClass reference the cluster name
    # explicitly (for subnet/SG tag discovery) AND the exact node
    # IAM role name (the ai-on-eks base module appends a
    # Terraform-generated suffix so the name isn't predictable).
    # Render both from the live cluster state rather than
    # requiring hand-editing of manifests.
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

    echo ""
    echo "=== Verifying installation ==="
    kubectl get runtimeclasses
    kubectl get nodepools 2>/dev/null || true
    kubectl get sandboxtemplates -A 2>/dev/null || true
}

install_kro() {
    echo ""
    echo "=== Phase 4 (optional): Installing KRO + AgentSandbox RGD ==="
    bash "$SCRIPT_DIR/manifests/kro-install.sh"
    kubectl apply -f "$SCRIPT_DIR/manifests/rgd-agent-sandbox.yaml"
    # Wait for the RGD's generated CRD to register so that users can
    # immediately kubectl apply the instance YAML.
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
    echo "  - Install an egress blueprint:"
    echo "      Standard EKS:  cd ../agent-egress-chained && ./install.sh"
    echo "      EKS Auto Mode: cd ../agent-egress-native  && ./install.sh"
    echo "  - Deploy the reference agent: kubectl apply -f $SCRIPT_DIR/manifests/sandbox-agent.yaml"
    echo "  - Run end-to-end conformance: ./conformance.sh"
    echo "  - Cleanup:                    cd terraform/_LOCAL && ./cleanup.sh"
}

case "$PHASE" in
    cluster)
        install_cluster
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
        install_sandbox
        install_manifests
        finish_message
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: cluster | sandbox | manifests | kro | all"
        exit 1
        ;;
esac
