#!/bin/bash
# Agent-with-Tools blueprint — mode-aware install.
#
# Deploys the full "sandbox as a tool" stack:
#   - Platform prerequisites (namespace, RuntimeClass, gVisor NodePool)
#   - Sandbox templates (code-exec + jupyter, tier-appropriate)
#   - Agent orchestrator (unsandboxed, with Bedrock IRSA)
#   - OpenWebUI (unsandboxed, user-facing chat UI)
#   - Egress policies (mode-aware: Cilium or ANP)
#
# Prerequisites:
#   - infra/agent-sandbox provisioned (EKS cluster + Karpenter + controller)
#   - egress example applied (provides base enforcement + Bedrock IRSA role):
#       cd ../egress && ./install.sh
#   - kubectl configured against the cluster
#
# Usage:
#   cd blueprints/agent-sandbox/agent-with-tools
#   ./install.sh                    # Full deploy (auto-resolves BEDROCK_ROLE_ARN)
#   ./install.sh uninstall          # Remove all blueprint resources

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_DIR="$(cd "$SCRIPT_DIR/../../../infra/agent-sandbox" && pwd)"
MANIFEST_DIR="$SCRIPT_DIR/manifests"
AGENT_DIR="$SCRIPT_DIR/agent"
NS="agent-sandboxes"
PHASE="${1:-install}"

# Configurable defaults
BEDROCK_MODEL_ID="${BEDROCK_MODEL_ID:-us.anthropic.claude-sonnet-4-20250514-v1:0}"

# Region precedence (matches sibling blueprints):
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
    REGION=$(kubectl config current-context 2>/dev/null \
        | awk -F':' '{print $4}' || echo "")
    REGION="${REGION:-us-west-2}"
fi
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

detect_compute_mode() {
    local enabled attempt
    for attempt in 1 2 3; do
        enabled=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
            --query 'cluster.computeConfig.enabled' --output text 2>/dev/null || echo "")
        if [ "$enabled" = "True" ] || [ "$enabled" = "true" ]; then
            COMPUTE_MODE="automode"
            SANDBOX_TIER="runc"
            log "Detected EKS Auto Mode — using runc tier for sandboxes."
            return 0
        fi
        if [ "$enabled" = "False" ] || [ "$enabled" = "false" ]; then
            COMPUTE_MODE="standard"
            SANDBOX_TIER="gvisor"
            log "Detected Standard EKS — using gVisor tier for sandboxes."
            return 0
        fi
        if [ "$attempt" -lt 3 ]; then sleep 2; fi
    done
    fail "Could not determine cluster compute mode after 3 attempts."
}

require_prereqs() {
    log "Checking prerequisites..."
    kubectl cluster-info >/dev/null 2>&1 \
        || fail "kubectl cannot reach the cluster"
    kubectl -n agent-sandbox-system get deployment agent-sandbox-controller >/dev/null 2>&1 \
        || fail "agent-sandbox controller missing — run infra/agent-sandbox/install.sh first"

    if [ -z "${BEDROCK_ROLE_ARN:-}" ]; then
        local default_role="${CLUSTER_NAME}-bedrock-irsa"
        BEDROCK_ROLE_ARN=$(aws iam get-role --role-name "$default_role" \
            --query 'Role.Arn' --output text 2>/dev/null || echo "")
        if [ -z "$BEDROCK_ROLE_ARN" ]; then
            fail "BEDROCK_ROLE_ARN not set and default role '$default_role' not found. Run ../egress/install.sh first or export BEDROCK_ROLE_ARN."
        fi
        log "Resolved BEDROCK_ROLE_ARN from default: $BEDROCK_ROLE_ARN"
    fi
}

ensure_platform_manifests() {
    # Ensure the namespace exists
    if ! kubectl get ns "$NS" >/dev/null 2>&1; then
        log "Creating namespace $NS..."
        kubectl apply -f "$INFRA_DIR/manifests/namespace.yaml"
    fi

    # For Standard EKS: ensure gVisor RuntimeClass + NodePool exist
    if [ "$COMPUTE_MODE" = "standard" ]; then
        if ! kubectl get runtimeclass gvisor >/dev/null 2>&1; then
            log "Applying gVisor RuntimeClass..."
            kubectl apply -f "$INFRA_DIR/manifests/runtimeclass-gvisor.yaml"
        fi

        if ! kubectl get nodepool agent-sandbox-gvisor >/dev/null 2>&1; then
            log "Applying gVisor Karpenter NodePool..."
            local karpenter_role
            karpenter_role=$(kubectl get ec2nodeclass m6i-cpu -o jsonpath='{.spec.role}' 2>/dev/null || echo "")
            if [ -z "$karpenter_role" ]; then
                log "WARNING: Could not resolve Karpenter node role — gVisor NodePool not applied."
                log "         Apply manually: sed -e 's|__CLUSTER_NAME__|$CLUSTER_NAME|g' -e 's|__KARPENTER_NODE_ROLE__|<role>|g' $INFRA_DIR/manifests/karpenter-nodepool-gvisor.yaml | kubectl apply -f -"
            else
                sed -e "s|__CLUSTER_NAME__|$CLUSTER_NAME|g" \
                    -e "s|__KARPENTER_NODE_ROLE__|$karpenter_role|g" \
                    "$INFRA_DIR/manifests/karpenter-nodepool-gvisor.yaml" \
                    | kubectl apply -f -
            fi
        fi
    fi
}

ensure_irsa_trust_policy() {
    # Ensure agent-orchestrator-sa is in the IRSA trust policy
    local role_name
    role_name=$(echo "$BEDROCK_ROLE_ARN" | awk -F'/' '{print $NF}')
    local current_trust
    current_trust=$(aws iam get-role --role-name "$role_name" \
        --query 'Role.AssumeRolePolicyDocument' --output json 2>/dev/null || echo "")

    if echo "$current_trust" | grep -q "agent-orchestrator-sa"; then
        log "IRSA trust policy already includes agent-orchestrator-sa."
        return 0
    fi

    log "Updating IRSA trust policy to include agent-orchestrator-sa..."
    local oidc_issuer oidc_id account_id
    oidc_issuer=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" \
        --query 'cluster.identity.oidc.issuer' --output text)
    oidc_id=$(echo "$oidc_issuer" | awk -F'/' '{print $NF}')
    account_id=$(aws sts get-caller-identity --query Account --output text)

    local trust_policy
    trust_policy=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${account_id}:oidc-provider/oidc.eks.${REGION}.amazonaws.com/id/${oidc_id}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.${REGION}.amazonaws.com/id/${oidc_id}:sub": [
            "system:serviceaccount:agent-sandboxes:sandbox-agent-sa",
            "system:serviceaccount:agent-sandboxes:composed-sandbox",
            "system:serviceaccount:agent-sandboxes:agent-orchestrator-sa"
          ],
          "oidc.eks.${REGION}.amazonaws.com/id/${oidc_id}:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
EOF
)
    echo "$trust_policy" > /tmp/agent-with-tools-trust.json
    aws iam update-assume-role-policy --role-name "$role_name" \
        --policy-document file:///tmp/agent-with-tools-trust.json
    rm -f /tmp/agent-with-tools-trust.json
    log "IRSA trust policy updated."
}

apply_sandbox_templates() {
    log "Applying sandbox templates (tier: $SANDBOX_TIER)..."
    kubectl apply -f "$MANIFEST_DIR/sandbox-code-exec-${SANDBOX_TIER}.yaml"
    kubectl apply -f "$MANIFEST_DIR/sandbox-jupyter-${SANDBOX_TIER}.yaml"

    # On Standard EKS, also apply runc variants as fallback
    if [ "$COMPUTE_MODE" = "standard" ]; then
        kubectl apply -f "$MANIFEST_DIR/sandbox-code-exec-runc.yaml"
        kubectl apply -f "$MANIFEST_DIR/sandbox-jupyter-runc.yaml"
    fi
}

apply_egress_policies() {
    log "Applying egress policies ($COMPUTE_MODE mode)..."
    if [ "$COMPUTE_MODE" = "automode" ]; then
        kubectl apply -f "$MANIFEST_DIR/egress/anp-agent-allowlist.yaml"
        kubectl apply -f "$MANIFEST_DIR/egress/anp-sandbox-allowlist.yaml"
    else
        kubectl apply -f "$MANIFEST_DIR/egress/cilium-agent-allowlist.yaml"
        kubectl apply -f "$MANIFEST_DIR/egress/cilium-sandbox-allowlist.yaml"
        kubectl apply -f "$MANIFEST_DIR/egress/cilium-openwebui-allowlist.yaml"
    fi
}

deploy_agent() {
    log "Deploying agent-orchestrator (model: $BEDROCK_MODEL_ID, region: $REGION)..."

    # Apply the deployment manifest first (creates SA, Role, RoleBinding, Service)
    sed -e "s|__AWS_REGION__|$REGION|g" \
        -e "s|__BEDROCK_MODEL_ID__|$BEDROCK_MODEL_ID|g" \
        -e "s|__SANDBOX_TIER__|$SANDBOX_TIER|g" \
        "$MANIFEST_DIR/agent-deployment.yaml" \
        | kubectl apply -f -

    # Overwrite the ConfigMap with real code (the manifest creates a placeholder;
    # this ensures the real files take precedence regardless of apply order)
    log "Injecting agent code into ConfigMap..."
    kubectl -n "$NS" create configmap agent-server-code \
        --from-file=agent_server.py="$AGENT_DIR/agent_server.py" \
        --from-file=tools.py="$AGENT_DIR/tools.py" \
        --from-file=requirements.txt="$AGENT_DIR/requirements.txt" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Annotate the ServiceAccount with IRSA
    kubectl annotate serviceaccount agent-orchestrator-sa -n "$NS" \
        "eks.amazonaws.com/role-arn=$BEDROCK_ROLE_ARN" \
        --overwrite

    # Restart the deployment so it picks up the real ConfigMap contents
    kubectl -n "$NS" rollout restart deployment/agent-orchestrator 2>/dev/null || true

    log "Waiting for agent-orchestrator to be ready (up to 5 min)..."
    kubectl -n "$NS" rollout status deployment/agent-orchestrator --timeout=300s || {
        log "WARNING: Agent deployment not ready within 5 min."
        log "  Check logs: kubectl -n $NS logs deployment/agent-orchestrator -c agent"
        log "  Common causes:"
        log "    - PyPI unreachable (check egress policy)"
        log "    - IRSA not working (check trust policy includes agent-orchestrator-sa)"
    }
}

deploy_openwebui() {
    log "Deploying OpenWebUI..."
    kubectl apply -f "$MANIFEST_DIR/openwebui-deployment.yaml"

    log "Waiting for OpenWebUI to be ready (up to 5 min — first boot downloads models)..."
    kubectl -n "$NS" rollout status deployment/openwebui --timeout=300s || {
        log "WARNING: OpenWebUI not ready within 5 min."
        log "  Check logs: kubectl -n $NS logs deployment/openwebui"
        log "  Common cause: egress policy blocking HuggingFace model download."
        log "  Verify: kubectl -n $NS get ciliumnetworkpolicy openwebui-allowlist"
    }
}

print_success() {
    echo ""
    log "=== Installation complete ==="
    echo ""
    echo "Compute mode:    $COMPUTE_MODE"
    echo "Sandbox tier:    $SANDBOX_TIER"
    echo "Bedrock model:   $BEDROCK_MODEL_ID"
    echo "Bedrock region:  $REGION"
    echo "IRSA role:       $BEDROCK_ROLE_ARN"
    echo ""
    echo "Access OpenWebUI:"
    local endpoint
    endpoint=$(kubectl -n "$NS" get svc openwebui -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    if [ -n "$endpoint" ]; then
        echo "  http://$endpoint:8080"
    else
        echo "  kubectl -n $NS port-forward svc/openwebui 8080:8080"
        echo "  Then open http://localhost:8080"
    fi
    echo ""
    echo "Verify agent health:"
    echo "  kubectl -n $NS exec deploy/agent-orchestrator -c agent -- python -c \"import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())\""
    echo ""
    echo "Test tool-calling:"
    echo "  kubectl -n $NS exec deploy/agent-orchestrator -c agent -- python -c \""
    echo "  import urllib.request, json"
    echo "  data = json.dumps({'messages':[{'role':'user','content':'Use code execution to print 2**10'}]}).encode()"
    echo "  req = urllib.request.Request('http://localhost:8000/v1/chat/completions', data=data, headers={'Content-Type':'application/json'})"
    echo "  resp = urllib.request.urlopen(req, timeout=300)"
    echo "  print(json.loads(resp.read().decode())['choices'][0]['message']['content'])"
    echo "  \""
    echo ""
    echo "Run conformance test:"
    echo "  ./conformance.sh"
}

uninstall() {
    log "Uninstalling agent-with-tools blueprint..."

    # Remove deployments + services
    kubectl -n "$NS" delete deployment agent-orchestrator --ignore-not-found
    kubectl -n "$NS" delete deployment openwebui --ignore-not-found
    kubectl -n "$NS" delete service agent-orchestrator --ignore-not-found
    kubectl -n "$NS" delete service openwebui --ignore-not-found
    kubectl -n "$NS" delete pvc openwebui-data --ignore-not-found
    kubectl -n "$NS" delete configmap agent-server-code --ignore-not-found
    kubectl -n "$NS" delete serviceaccount agent-orchestrator-sa --ignore-not-found
    kubectl -n "$NS" delete role agent-orchestrator-role --ignore-not-found
    kubectl -n "$NS" delete rolebinding agent-orchestrator-binding --ignore-not-found

    # Remove sandbox templates
    kubectl delete -f "$MANIFEST_DIR/sandbox-code-exec-runc.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "$MANIFEST_DIR/sandbox-code-exec-gvisor.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "$MANIFEST_DIR/sandbox-jupyter-runc.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "$MANIFEST_DIR/sandbox-jupyter-gvisor.yaml" --ignore-not-found 2>/dev/null || true

    # Remove egress policies
    kubectl delete -f "$MANIFEST_DIR/egress/cilium-agent-allowlist.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "$MANIFEST_DIR/egress/cilium-sandbox-allowlist.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "$MANIFEST_DIR/egress/cilium-openwebui-allowlist.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "$MANIFEST_DIR/egress/anp-agent-allowlist.yaml" --ignore-not-found 2>/dev/null || true
    kubectl delete -f "$MANIFEST_DIR/egress/anp-sandbox-allowlist.yaml" --ignore-not-found 2>/dev/null || true

    # Remove any lingering sandbox pods created by the agent
    kubectl -n "$NS" delete sandboxclaims -l agent-sandbox/managed-by=agent-with-tools --ignore-not-found 2>/dev/null || true

    log "Uninstall complete."
}

case "$PHASE" in
    install)
        detect_compute_mode
        require_prereqs
        ensure_platform_manifests
        ensure_irsa_trust_policy
        apply_egress_policies
        apply_sandbox_templates
        deploy_agent
        deploy_openwebui
        print_success
        ;;
    uninstall)
        uninstall
        ;;
    *)
        echo "Unknown phase: $PHASE" >&2
        echo "Usage: $0 [install|uninstall]" >&2
        exit 1
        ;;
esac
