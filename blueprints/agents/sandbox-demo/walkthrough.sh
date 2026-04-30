#!/bin/bash
# Agent Sandbox on EKS — 15-minute showcase walkthrough script.
#
# Preconditions (all done before the showcase starts):
#   - Cluster provisioned:      ./install.sh cluster   (20-30 min)
#   - Cilium + Hubble up:       ./install.sh cilium    (3-5 min)
#   - agent-sandbox controller: ./install.sh sandbox   (1-2 min)
#   - Manifests applied:        ./install.sh manifests (30s)
#   - Pod Identity association seeded for sandbox-demo-agent SA →
#     IAM role with bedrock:InvokeModel on the demo model
#   - ConfigMap + Sandbox applied with the actual agent.py contents
#     (handled by this script's `setup` phase if not yet done)
#   - Hubble UI port-forward running in a separate terminal
#   - Pre-warmed gVisor node exists (first sandbox provision is slow
#     due to gVisor shim install — trigger it 5 min before showcase)
#
# Usage during showcase:
#   ./walkthrough.sh setup     # (one-time, before showcase starts)
#   ./walkthrough.sh run       # (live, during the 15 minutes)
#   ./walkthrough.sh cleanup   # (between rehearsals; NOT during live)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_DIR="$SCRIPT_DIR/../../../infra/agent-sandbox"
NS="agent-sandboxes"
PHASE="${1:-run}"

pause() {
    # During rehearsals, pausing on ENTER lets the narrator pace the
    # demo. During live, the operator presses ENTER when the verbal
    # beat hits. Set PAUSE=0 env var to auto-run through everything.
    if [ "${PAUSE:-1}" = "1" ]; then
        echo ""
        echo "    [press ENTER to continue]"
        read -r
    else
        sleep 2
    fi
}

banner() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
}

setup_agent_configmap() {
    # ConfigMap needs the actual agent.py contents. Recreate on every
    # setup run so script edits land without confusion.
    kubectl -n $NS delete configmap sandbox-demo-agent-script --ignore-not-found
    kubectl -n $NS create configmap sandbox-demo-agent-script \
        --from-file=agent.py="$SCRIPT_DIR/agent.py"
}

setup_pod_identity_association() {
    # EKS Pod Identity wiring — creates the association between the
    # sandbox-demo-agent ServiceAccount and the IAM role that grants
    # bedrock:InvokeModel. The IAM role itself is expected to exist
    # (created by Terraform or pre-provisioned via the AWS console).
    #
    # Environment variables expected:
    #   CLUSTER_NAME      — EKS cluster name (default: agent-sandbox)
    #   BEDROCK_ROLE_ARN  — IAM role ARN with bedrock:InvokeModel
    CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"
    REGION="${AWS_REGION:-us-east-1}"

    if [ -z "${BEDROCK_ROLE_ARN:-}" ]; then
        echo "WARNING: BEDROCK_ROLE_ARN not set — skipping Pod Identity assoc."
        echo "         The Bedrock call step will fail with a credentials error."
        echo "         Create the IAM role, then set BEDROCK_ROLE_ARN and rerun setup."
        return 0
    fi

    # Check if an association already exists (setup is idempotent).
    existing=$(aws eks list-pod-identity-associations \
        --cluster-name "$CLUSTER_NAME" \
        --namespace "$NS" \
        --service-account sandbox-demo-agent \
        --region "$REGION" \
        --query 'associations[0].associationId' \
        --output text 2>/dev/null || true)

    if [ -n "$existing" ] && [ "$existing" != "None" ]; then
        echo "Pod Identity association already exists: $existing"
        return 0
    fi

    aws eks create-pod-identity-association \
        --cluster-name "$CLUSTER_NAME" \
        --namespace "$NS" \
        --service-account sandbox-demo-agent \
        --role-arn "$BEDROCK_ROLE_ARN" \
        --region "$REGION"
    echo "Pod Identity association created."
}

phase_setup() {
    banner "Setup: ConfigMap + Pod Identity + apply Sandbox"
    setup_agent_configmap
    setup_pod_identity_association
    kubectl apply -f "$INFRA_DIR/manifests/demo-agent.yaml"

    echo ""
    echo "Waiting for the Sandbox pod to become Ready (may take 2-3 min on"
    echo "first provisioning while Karpenter spins up a gVisor node)..."
    kubectl -n $NS wait --for=condition=Ready pod/sandbox-demo --timeout=300s || {
        echo ""
        echo "Pod did not become Ready in 5 min. Debug with:"
        echo "  kubectl -n $NS describe pod sandbox-demo"
        echo "  kubectl get nodepools"
        echo "  kubectl get nodes -l agent-sandbox/runtime=gvisor"
        exit 1
    }
    echo ""
    echo "Sandbox ready. Run './walkthrough.sh run' for the showcase."
}

phase_run() {
    banner "Act 1: The sandbox is already running, let's look at it"
    echo "Command: kubectl get sandbox -n agent-sandboxes"
    kubectl get sandbox -n $NS
    echo ""
    echo "Command: kubectl get pods -n agent-sandboxes -o wide"
    kubectl get pods -n $NS -o wide
    pause

    banner "Act 2: Confirm this is actually running on gVisor"
    echo "Command: kubectl describe pod sandbox-demo -n agent-sandboxes | grep -E 'Runtime|Node:'"
    kubectl describe pod sandbox-demo -n $NS | grep -E "Runtime|Node:"
    echo ""
    echo "Command: kubectl get node \$(kubectl get pod sandbox-demo -n $NS -o jsonpath='{.spec.nodeName}') --show-labels"
    kubectl get node $(kubectl get pod sandbox-demo -n $NS -o jsonpath='{.spec.nodeName}') --show-labels | tr , '\n'
    pause

    banner "Act 3: Show the network policies that are enforced"
    echo "Cluster-wide (admin tier — blocks IMDS + link-local for every pod):"
    echo "Command: kubectl get ciliumclusterwidenetworkpolicies"
    kubectl get ciliumclusterwidenetworkpolicies
    echo ""
    echo "Namespace (app tier — per-sandbox FQDN allowlist):"
    echo "Command: kubectl get ciliumnetworkpolicies -n agent-sandboxes"
    kubectl get ciliumnetworkpolicies -n $NS
    pause

    banner "Act 4: Run the reference agent inside the sandbox"
    echo "This walks through four steps:"
    echo "  1. pip install boto3 from PyPI       — should PASS (PyPI allowed)"
    echo "  2. Call Bedrock Claude               — should PASS (Bedrock allowed)"
    echo "  3. Execute the model-generated code  — should PASS (syscalls via Sentry)"
    echo "  4. curl demo-blocked.example.com     — should BLOCK (not on allowlist)"
    echo ""
    echo "Watch Hubble UI during the run (http://localhost:12000)"
    pause
    echo "Command: kubectl exec -n agent-sandboxes sandbox-demo -c agent-runtime -- python /workspace/agent.py"
    echo ""
    kubectl exec -n $NS sandbox-demo -c agent-runtime -- python /workspace/agent.py
    pause

    banner "Act 5: Recap via Hubble flow map"
    echo "In Hubble UI, filter to namespace=agent-sandboxes"
    echo "Expected flows:"
    echo "  - ALLOWED to bedrock-runtime.us-east-1.amazonaws.com:443"
    echo "  - ALLOWED to pypi.org:443 + files.pythonhosted.org:443"
    echo "  - DROPPED to demo-blocked.example.com:443 (policy-denied)"
    echo ""
    echo "End of live demo. Switch to Phase 2+3 preview slides."
}

phase_kro_demo() {
    banner "Stretch: KRO AgentSandbox composite CRD"
    echo "The blueprint ships a kro ResourceGraphDefinition that turns"
    echo "this three-manifest demo (ServiceAccount + Sandbox + CiliumNetworkPolicy)"
    echo "into a single-CR AgentSandbox abstraction. Customers who adopt it"
    echo "write one YAML instead of three."
    echo ""
    echo "Command: kubectl get rgd"
    kubectl get rgd 2>&1 | head -5
    echo ""
    pause
    echo "The RGD defines an AgentSandbox CRD with runtimeClass + iamRoleArn"
    echo "+ image + command fields. Here's the one-CR instance:"
    echo ""
    echo "Command: kubectl get agentsandbox -n agent-sandboxes -o yaml"
    kubectl get agentsandbox demo-composed -n $NS -o yaml 2>&1 | head -30
    pause
    echo "...and the child resources KRO composes from it:"
    echo ""
    echo "Command: kubectl get sandbox,serviceaccount demo-composed -n agent-sandboxes"
    kubectl get sandbox,serviceaccount demo-composed -n $NS 2>&1
    echo ""
    echo "Same gVisor runtime, same IRSA wiring, same egress policy"
    echo "(the CiliumNetworkPolicy is a sibling that matches via label)"
    echo "— but written as ~15 lines of YAML instead of ~150."
}

phase_cleanup() {
    banner "Cleanup (between rehearsals only — not during live demo)"
    kubectl delete -f "$INFRA_DIR/manifests/demo-agent.yaml" --ignore-not-found
    kubectl -n $NS delete configmap sandbox-demo-agent-script --ignore-not-found
    echo ""
    echo "Sandbox + ConfigMap removed. Pod Identity association kept."
    echo "Re-run './walkthrough.sh setup' to re-provision for next run."
}

case "$PHASE" in
    setup)    phase_setup    ;;
    run)      phase_run      ;;
    kro)      phase_kro_demo ;;
    cleanup)  phase_cleanup  ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: setup | run | kro | cleanup"
        exit 1
        ;;
esac
