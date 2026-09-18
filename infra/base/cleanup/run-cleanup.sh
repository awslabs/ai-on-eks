#!/bin/bash
# AI on EKS — shared cleanup driver.
#
# Generalizes the hierarchical teardown that proved out in
# infra/agent-sandbox/cleanup.sh into a reusable driver. Each component calls
# this with its cluster/region/local-dir and an optional phase-0 hook for
# component-specific pre-teardown (e.g., egress example uninstall). Seeds #334.
#
# Phases (each idempotent; tolerates being skipped after a partial destroy):
#   0. Component phase-0 hook (optional) — component-specific pre-teardown.
#   1. Karpenter scale-down — stop new node launches mid-teardown.
#   2. Karpenter finalizer drop — EC2NodeClass + NodePool.
#   3. Karpenter instance termination — release ENIs blocking subnet delete.
#   4. Base terraform destroy — retry-and-verify against live AWS state.
#   5. Auxiliary AWS resource sweep — cluster SGs, placement groups, KMS
#      aliases, CloudWatch log groups.
#
# Inputs (env):
#   CLUSTER_NAME   (required) cluster name / resource tag value
#   REGION         (required) AWS region
#   LOCAL_DIR      (required) terraform working dir with the base copy + cleanup.sh
#   PHASE0_HOOK    (optional) command run as phase 0 (e.g. "/path/install.sh uninstall")
#
# Exit 0 on full teardown, 1 if phase 4 did not complete after retries.

set -euo pipefail

: "${CLUSTER_NAME:?CLUSTER_NAME required}"
: "${REGION:?REGION required}"
: "${LOCAL_DIR:?LOCAL_DIR required}"
PHASE0_HOOK="${PHASE0_HOOK:-}"

echo "Cleanup driver: cluster=$CLUSTER_NAME region=$REGION local_dir=$LOCAL_DIR"

echo ""
echo "=== Phase 0: Component pre-teardown hook ==="
if [ -n "$PHASE0_HOOK" ]; then
    echo "  Running: $PHASE0_HOOK"
    # Best-effort — component hooks are idempotent and tolerate a partial state.
    bash -c "$PHASE0_HOOK" || true
else
    echo "  No phase-0 hook provided — skipping."
fi

echo ""
echo "=== Phase 1: Scale Karpenter controller to 0 (stop new node launches) ==="
if kubectl -n kube-system get deployment karpenter >/dev/null 2>&1; then
    echo "  Scaling karpenter deployment to 0 replicas..."
    kubectl -n kube-system scale deployment karpenter --replicas=0 >/dev/null 2>&1 || true
    sleep 10
else
    echo "  Karpenter deployment not present — skipping (cluster already partly destroyed)."
fi

echo ""
echo "=== Phase 2: Drop Karpenter finalizers on EC2NodeClass + NodePool ==="
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
echo "=== Phase 3: Terminate any Karpenter-provisioned EC2 instances ==="
# Primary filter: instances carrying the karpenter nodepool tag AND this
# cluster's eks cluster-name tag (multi-cluster-safe).
KARPENTER_INSTANCES=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag-key,Values=karpenter.sh/nodepool" \
              "Name=tag:aws:eks:cluster-name,Values=$CLUSTER_NAME" \
              "Name=instance-state-name,Values=running,pending,stopping" \
    --query "Reservations[].Instances[].InstanceId" \
    --output text 2>/dev/null || echo "")

# Secondary sweep: the aws:eks:cluster-name tag has observed
# eventual-consistency lag; fall back to NodePool-name prefix (<cluster>-*).
if [ -z "$KARPENTER_INSTANCES" ]; then
    echo "  Primary filter returned empty — running secondary NodePool-name sweep..."
    KARPENTER_INSTANCES=$(aws ec2 describe-instances \
        --region "$REGION" \
        --filters "Name=tag:karpenter.sh/nodepool,Values=${CLUSTER_NAME}-*" \
                  "Name=instance-state-name,Values=running,pending,stopping" \
        --query "Reservations[].Instances[].InstanceId" \
        --output text 2>/dev/null || echo "")
fi

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
echo "=== Phase 4: Run base module cleanup ==="
PHASE_4_SUCCESS=false
if [ -d "$LOCAL_DIR" ]; then
    # Remove in-cluster helm/kubectl resources from state first (except
    # ArgoCD, which the destroy needs to sweep its child Applications). They
    # die with the cluster; destroying them through a flaky cluster API stalls.
    pushd "$LOCAL_DIR" >/dev/null
    for stale_resource in $(terraform state list 2>/dev/null \
            | grep -E '^helm_release\.|^kubectl_manifest\.' || true); do
        if [[ "$stale_resource" == *"argocd"* ]]; then continue; fi
        echo "  Removing $stale_resource from state (cluster destroy will sweep)"
        terraform state rm "$stale_resource" >/dev/null 2>&1 || true
    done
    popd >/dev/null

    for attempt in 1 2 3; do
        echo ""
        echo "  Base destroy attempt $attempt..."
        ( cd "$LOCAL_DIR" && bash ./cleanup.sh ) || true

        remaining_vpc=$(aws ec2 describe-vpcs --region "$REGION" \
            --filters "Name=tag:Name,Values=${CLUSTER_NAME}" \
            --query "Vpcs[].VpcId" --output text 2>/dev/null || echo "")
        remaining_cluster=$(aws eks describe-cluster --name "$CLUSTER_NAME" \
            --region "$REGION" --query 'cluster.status' --output text 2>/dev/null || echo "")
        if [ -z "$remaining_vpc" ] && [ -z "$remaining_cluster" ]; then
            echo "  Base destroy succeeded — no VPC or cluster remaining."
            PHASE_4_SUCCESS=true
            break
        fi

        echo "  Resources still present (vpc='$remaining_vpc' cluster='$remaining_cluster') — retrying with raw terraform destroy..."
        ( cd "$LOCAL_DIR" && terraform destroy -auto-approve -var-file=../blueprint.tfvars ) || true

        remaining_vpc=$(aws ec2 describe-vpcs --region "$REGION" \
            --filters "Name=tag:Name,Values=${CLUSTER_NAME}" \
            --query "Vpcs[].VpcId" --output text 2>/dev/null || echo "")
        remaining_cluster=$(aws eks describe-cluster --name "$CLUSTER_NAME" \
            --region "$REGION" --query 'cluster.status' --output text 2>/dev/null || echo "")
        if [ -z "$remaining_vpc" ] && [ -z "$remaining_cluster" ]; then
            echo "  Raw destroy retry succeeded."
            PHASE_4_SUCCESS=true
            break
        fi
        echo "  Resources still present after retry; will try again next attempt."
    done

    if [ "$PHASE_4_SUCCESS" != "true" ]; then
        echo ""
        echo "  WARNING: Phase 4 did not fully complete after 3 attempts."
        echo "  Phase 5 will run to clean up known auxiliary resources, but"
        echo "  manual inspection of remaining VPC/EKS resources is required."
    fi
else
    echo "  $LOCAL_DIR not present — skipping base destroy (already complete)."
    PHASE_4_SUCCESS=true
fi

echo ""
echo "=== Phase 5: Sweep auxiliary AWS resources ==="
LINGERING_VPC=$(aws ec2 describe-vpcs --region "$REGION" \
    --filters "Name=tag:Name,Values=${CLUSTER_NAME}" \
    --query "Vpcs[].VpcId" --output text 2>/dev/null || echo "")
if [ -n "$LINGERING_VPC" ]; then
    echo "  VPC $LINGERING_VPC still present — sweeping non-default security groups..."
    SG_IDS=$(aws ec2 describe-security-groups --region "$REGION" \
        --filters "Name=vpc-id,Values=$LINGERING_VPC" \
        --query "SecurityGroups[?GroupName!='default'].GroupId" --output text 2>/dev/null || echo "")
    for sg in $SG_IDS; do
        echo "    Deleting security group $sg"
        aws ec2 delete-security-group --region "$REGION" --group-id "$sg" >/dev/null 2>&1 || true
    done
    if [ -d "$LOCAL_DIR" ]; then
        echo "  Retrying VPC destroy via terraform..."
        ( cd "$LOCAL_DIR" && terraform destroy -auto-approve -var-file=../blueprint.tfvars -target=module.vpc ) || true
    fi
fi

echo "  Placement groups:"
PG_NAMES=$(aws ec2 describe-placement-groups --region "$REGION" \
    --filters "Name=group-name,Values=${CLUSTER_NAME}-*" \
    --query "PlacementGroups[].GroupName" --output text 2>/dev/null || echo "")
if [ -n "$PG_NAMES" ]; then
    for pg in $PG_NAMES; do
        echo "    Deleting $pg"
        aws ec2 delete-placement-group --region "$REGION" --group-name "$pg" >/dev/null 2>&1 || true
    done
else
    echo "    None found."
fi

echo "  KMS aliases:"
KMS_ALIASES=$(aws kms list-aliases --region "$REGION" \
    --query "Aliases[?AliasName=='alias/eks/${CLUSTER_NAME}'].AliasName" --output text 2>/dev/null || echo "")
if [ -n "$KMS_ALIASES" ]; then
    for alias in $KMS_ALIASES; do
        echo "    Deleting $alias"
        aws kms delete-alias --region "$REGION" --alias-name "$alias" >/dev/null 2>&1 || true
    done
else
    echo "    None found."
fi

echo "  CloudWatch log groups:"
LOG_GROUPS=$(aws logs describe-log-groups --region "$REGION" \
    --log-group-name-prefix "/aws/eks/${CLUSTER_NAME}" \
    --query "logGroups[].logGroupName" --output text 2>/dev/null || echo "")
if [ -n "$LOG_GROUPS" ]; then
    for lg in $LOG_GROUPS; do
        echo "    Deleting $lg"
        aws logs delete-log-group --region "$REGION" --log-group-name "$lg" >/dev/null 2>&1 || true
    done
else
    echo "    None found."
fi

echo ""
if [ "$PHASE_4_SUCCESS" = "true" ]; then
    echo "=== Cleanup complete ==="
else
    echo "=== Cleanup partially complete — Phase 4 did not finish after 3 retries ==="
    echo "Manual: cd $LOCAL_DIR && terraform destroy -auto-approve -var-file=../blueprint.tfvars"
    exit 1
fi
