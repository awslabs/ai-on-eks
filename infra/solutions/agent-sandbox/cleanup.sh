#!/bin/bash
# Agent Sandbox solution — hierarchical teardown.
#
# Wraps the base module's cleanup.sh with pre/post phases that the
# base flow doesn't know about:
#
#   0. Pre-cleanup: invoke whichever egress example is installed
#      (chained or native) with its `uninstall` phase. Each example's
#      uninstall removes its own resources (CNPs/ANPs, Cilium for
#      chained) AND the Bedrock IRSA role it provisioned. Skipped
#      silently if the example was never installed.
#
#   1. Pre-cleanup: drop Karpenter finalizers on EC2NodeClass +
#      NodePool resources before the base destroy starts. Karpenter's
#      finalizer (`karpenter.k8s.aws/termination`) waits for the
#      controller to drain managed instances; once the EKS cluster
#      is being torn down, the controller pod is unschedulable and
#      finalizers stall indefinitely. Dropping finalizers up front
#      lets the base destroy walk through cleanly.
#
#   2. Pre-cleanup: terminate any lingering Karpenter-provisioned
#      EC2 instances directly. Subnet deletion blocks on attached
#      ENIs; if Karpenter didn't get a chance to drain instances
#      before its controller was deleted, the instances outlive the
#      controller and block VPC teardown. Terminating them up front
#      releases the ENIs so the base destroy completes.
#
#   3. Base teardown: cd terraform/_LOCAL && ./cleanup.sh
#
#   4. Post-cleanup: sweep auxiliary AWS resources by tag. Terraform
#      should handle these on a clean destroy, but state-loss or
#      partial-destroy scenarios leave behind:
#        - EC2 placement groups (e.g., agent-sandbox-nvidia-gpu)
#        - KMS aliases (alias/eks/<cluster>)
#        - CloudWatch log groups (/aws/eks/<cluster>/cluster)
#      Walk each resource type by name prefix or tag and delete
#      idempotently. Safe to re-run.
#
# Usage:
#   cd infra/solutions/agent-sandbox
#   ./cleanup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_DIR="$SCRIPT_DIR/terraform/_LOCAL"

# Resolve cluster name + region from the live tfvars so the auxiliary
# sweep filters correctly. Fall back to defaults if the local copy
# isn't present (e.g., user is re-running cleanup after a partial
# destroy already removed terraform/_LOCAL).
if [ -f "$SCRIPT_DIR/terraform/blueprint.tfvars" ]; then
    CLUSTER_NAME=$(grep -E '^name\s*=' "$SCRIPT_DIR/terraform/blueprint.tfvars" | head -1 | awk -F'"' '{print $2}')
fi
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

echo "=== Phase 0: Run egress example uninstall (releases CNPs/ANPs + IRSA role) ==="
# Try each example's uninstall phase. Each one is idempotent — if
# the example wasn't installed, its uninstall is a no-op. We don't
# know which example the user picked, so try both. Failures are
# tolerated (best-effort cleanup).
for example_dir in "$SCRIPT_DIR/examples/agent-egress-chained" "$SCRIPT_DIR/examples/agent-egress-native"; do
    if [ -x "$example_dir/install.sh" ]; then
        echo "  Running $(basename "$example_dir") uninstall..."
        ( cd "$example_dir" && ./install.sh uninstall ) || true
    fi
done

echo ""
echo "=== Phase 1: Drop Karpenter finalizers on EC2NodeClass + NodePool ==="
# Best-effort — kubectl may already be unable to reach the cluster
# if a prior destroy partially completed. `|| true` keeps the script
# moving in that case.
if kubectl get ec2nodeclasses -o name >/dev/null 2>&1; then
    for nc in $(kubectl get ec2nodeclasses -o name 2>/dev/null); do
        echo "  Patching finalizer on $nc"
        kubectl patch "$nc" --type=merge -p '{"metadata":{"finalizers":[]}}' >/dev/null 2>&1 || true
    done
fi
if kubectl get nodepools -o name >/dev/null 2>&1; then
    for np in $(kubectl get nodepools -o name 2>/dev/null); do
        echo "  Patching finalizer on $np"
        kubectl patch "$np" --type=merge -p '{"metadata":{"finalizers":[]}}' >/dev/null 2>&1 || true
    done
fi

echo ""
echo "=== Phase 2: Terminate any Karpenter-provisioned EC2 instances ==="
# Filter by presence of the karpenter.sh/nodepool tag key. Compound
# tag filters with `Values=*` don't behave like a wildcard match in
# the EC2 API — `tag-key` is the right filter shape for "instances
# carrying this tag at all". Cross-reference against the cluster-name
# tag so we only terminate instances belonging to this deployment
# (multi-cluster accounts may have other agent-sandbox-flavored
# clusters running alongside).
KARPENTER_INSTANCES=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag-key,Values=karpenter.sh/nodepool" \
              "Name=tag:aws:eks:cluster-name,Values=$CLUSTER_NAME" \
              "Name=instance-state-name,Values=running,pending,stopping" \
    --query "Reservations[].Instances[].InstanceId" \
    --output text 2>/dev/null || echo "")
if [ -n "$KARPENTER_INSTANCES" ]; then
    echo "  Terminating: $KARPENTER_INSTANCES"
    # shellcheck disable=SC2086
    aws ec2 terminate-instances --region "$REGION" --instance-ids $KARPENTER_INSTANCES \
        --query "TerminatingInstances[].InstanceId" --output text >/dev/null
    echo "  Waiting for instances to terminate..."
    # shellcheck disable=SC2086
    aws ec2 wait instance-terminated --region "$REGION" --instance-ids $KARPENTER_INSTANCES || true
else
    echo "  No Karpenter-provisioned instances found."
fi

echo ""
echo "=== Phase 3: Run base module cleanup ==="
if [ -d "$LOCAL_DIR" ]; then
    # Pre-step: remove in-cluster helm/kubectl resources from state.
    # These resources live inside the cluster and will be deleted with
    # the cluster itself — but if terraform tries to destroy them
    # *through the cluster API* during the destroy run, the API may
    # already be flaky (helm provider reports "context deadline
    # exceeded" against karpenter-crd or similar). Removing them from
    # state first lets the base destroy walk through cleanly without
    # needing a healthy cluster API for each in-cluster resource.
    pushd "$LOCAL_DIR" >/dev/null
    for stale_resource in $(terraform state list 2>/dev/null \
            | grep -E '^helm_release\.|^kubectl_manifest\.' \
            || true); do
        # Skip ArgoCD itself — the destroy needs argocd to clean up
        # its child Application resources cleanly. Everything else
        # (kubectl_manifest.<addon>, helm_release.karpenter_crd, etc.)
        # is fair game for state-removal since the cluster destroy
        # will sweep them.
        if [[ "$stale_resource" == *"argocd"* ]]; then
            continue
        fi
        echo "  Removing $stale_resource from state (cluster destroy will sweep)"
        terraform state rm "$stale_resource" >/dev/null 2>&1 || true
    done
    popd >/dev/null

    # Run base cleanup in a subshell so its non-zero exits (e.g.,
    # `kubectl delete rayjob` against a cluster that has no Ray CRDs)
    # don't kill our wrapper's `set -e`. The base script was written
    # with best-effort semantics; we honor that here.
    ( cd "$LOCAL_DIR" && bash ./cleanup.sh ) || true
else
    echo "  $LOCAL_DIR not present — skipping base destroy (already complete)."
fi

echo ""
echo "=== Phase 4: Sweep auxiliary AWS resources ==="

# Placement groups — Terraform's destroy of an EKS managed node
# group with placement strategy doesn't always release these on
# eventual-consistency boundaries. Sweep by cluster-name prefix.
echo "  Placement groups:"
PG_NAMES=$(aws ec2 describe-placement-groups \
    --region "$REGION" \
    --filters "Name=group-name,Values=${CLUSTER_NAME}-*" \
    --query "PlacementGroups[].GroupName" \
    --output text 2>/dev/null || echo "")
if [ -n "$PG_NAMES" ]; then
    for pg in $PG_NAMES; do
        echo "    Deleting $pg"
        aws ec2 delete-placement-group --region "$REGION" --group-name "$pg" >/dev/null 2>&1 || true
    done
else
    echo "    None found."
fi

# KMS aliases — `alias/eks/<cluster>` is created by the EKS module's
# cluster_encryption block. Aliases sometimes stick around when the
# underlying key was scheduled for deletion but the alias detach
# didn't propagate.
echo "  KMS aliases:"
KMS_ALIASES=$(aws kms list-aliases \
    --region "$REGION" \
    --query "Aliases[?AliasName=='alias/eks/${CLUSTER_NAME}'].AliasName" \
    --output text 2>/dev/null || echo "")
if [ -n "$KMS_ALIASES" ]; then
    for alias in $KMS_ALIASES; do
        echo "    Deleting $alias"
        aws kms delete-alias --region "$REGION" --alias-name "$alias" >/dev/null 2>&1 || true
    done
else
    echo "    None found."
fi

# CloudWatch log groups — /aws/eks/<cluster>/cluster + any
# nested groups with the cluster prefix.
echo "  CloudWatch log groups:"
LOG_GROUPS=$(aws logs describe-log-groups \
    --region "$REGION" \
    --log-group-name-prefix "/aws/eks/${CLUSTER_NAME}" \
    --query "logGroups[].logGroupName" \
    --output text 2>/dev/null || echo "")
if [ -n "$LOG_GROUPS" ]; then
    for lg in $LOG_GROUPS; do
        echo "    Deleting $lg"
        aws logs delete-log-group --region "$REGION" --log-group-name "$lg" >/dev/null 2>&1 || true
    done
else
    echo "    None found."
fi

echo ""
echo "=== Cleanup complete ==="
echo ""
echo "Resources removed:"
echo "  - All terraform-managed resources (EKS, VPC, IAM, KMS, log groups)"
echo "  - Bedrock IRSA role (via the egress example's uninstall phase)"
echo "  - Karpenter-provisioned EC2 instances"
echo "  - Auxiliary AWS resources (placement groups, KMS aliases, log groups)"
