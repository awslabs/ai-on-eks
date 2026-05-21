#!/bin/bash
# Agent Sandbox solution — hierarchical teardown.
#
# Wraps the base module's cleanup.sh with pre/post phases that the
# base flow doesn't know about. Phases run in order; each is
# idempotent and tolerates being skipped if its prerequisites are
# missing (e.g., partial destroy already removed the cluster).
#
#   0. Egress example uninstall — removes any installed CNPs/ANPs +
#      Bedrock IRSA role provisioned by the egress example.
#   1. Karpenter scale-down — sets controller replicas to 0 so it
#      stops launching replacement nodes during teardown. Without
#      this, Phase 3's instance sweep hits a race where new nodes
#      come up faster than ENIs can release.
#   2. Karpenter finalizer drop — patches EC2NodeClass + NodePool
#      finalizers to empty. The `karpenter.k8s.aws/termination`
#      finalizer waits for the controller, which is unschedulable
#      once the cluster destroy starts, so finalizers stall
#      indefinitely without this step.
#   3. Karpenter instance termination — terminates lingering
#      Karpenter-launched EC2 instances directly. Subnet deletion
#      blocks on attached ENIs; releasing them up front lets the
#      base destroy walk through cleanly.
#   4. Base teardown — `terraform destroy` with retry-and-verify.
#      Re-runs up to 3 times if VPC or EKS cluster persist after
#      the destroy returns; verification queries AWS state directly
#      rather than trusting script exit codes.
#   5. Auxiliary AWS resource sweep — placement groups, KMS aliases,
#      CloudWatch log groups. Terraform handles these on a clean
#      destroy; this phase covers state-loss / partial-destroy cases.
#
# Usage:
#   cd infra/agent-sandbox
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
    # Region from tfvars takes precedence over env vars — the cluster
    # is always in the region the tfvars declared. The line is
    # commented out by default in the shipped tfvars (base module
    # default is us-west-2); pick that up if no override exists.
    TFVARS_REGION=$(grep -E '^region\s*=' "$SCRIPT_DIR/terraform/blueprint.tfvars" | head -1 | awk -F'"' '{print $2}' || echo "")
fi
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"
# Region precedence: tfvars > AWS_REGION env > AWS_DEFAULT_REGION env
# > kubectl context > base module default (us-west-2). Without this,
# Phase 3's instance sweep queries the wrong region and silently
# misses Karpenter-provisioned nodes, leaving them as orphans that
# block VPC delete in Phase 4.
if [ -n "${TFVARS_REGION:-}" ]; then
    REGION="$TFVARS_REGION"
elif [ -n "${AWS_REGION:-}" ]; then
    REGION="$AWS_REGION"
elif [ -n "${AWS_DEFAULT_REGION:-}" ]; then
    REGION="$AWS_DEFAULT_REGION"
else
    # Try to extract from kubectl context (format: arn:aws:eks:<region>:...)
    REGION=$(kubectl config current-context 2>/dev/null | awk -F':' '{print $4}' || echo "")
    REGION="${REGION:-us-west-2}"
fi
echo "Resolved cluster=$CLUSTER_NAME region=$REGION"

echo "=== Phase 0: Run egress example uninstall (releases CNPs/ANPs + IRSA role) ==="
# The agent-egress example's uninstall is idempotent and mode-aware
# — it auto-detects the cluster's compute mode and removes the
# matching policy backend (Cilium or ANP). Failures are tolerated
# (best-effort cleanup). Egress example moved to blueprints/ per the
# May 21 maintainer feedback restructure.
example_dir="$SCRIPT_DIR/../../blueprints/agent-sandbox/egress"
if [ -x "$example_dir/install.sh" ]; then
    echo "  Running agent-egress uninstall..."
    ( cd "$example_dir" && ./install.sh uninstall ) || true
fi

echo ""
echo "=== Phase 1: Scale Karpenter controller to 0 (stop new node launches) ==="
# Karpenter's reconcile loop is the only thing that provisions
# instances. Scaling it to 0 replicas halts the launch path before
# we attempt to terminate existing nodes — without this, Karpenter
# may spin up a replacement during Phase 3's wait, leaving a
# never-tagged orphan that blocks subnet deletion.
#
# Best-effort — if Karpenter isn't deployed (cluster already partly
# destroyed) the kubectl call fails harmlessly.
if kubectl -n kube-system get deployment karpenter >/dev/null 2>&1; then
    echo "  Scaling karpenter deployment to 0 replicas..."
    kubectl -n kube-system scale deployment karpenter --replicas=0 >/dev/null 2>&1 || true
    # Wait briefly so any in-flight launch decision can drain. Karpenter
    # commits launches via cloudprovider call before the reconcile
    # iteration completes, so most pending launches will fly even
    # after scale-down — that's fine, Phase 3 catches them.
    sleep 10
else
    echo "  Karpenter deployment not present — skipping (cluster already partly destroyed)."
fi

echo ""
echo "=== Phase 2: Drop Karpenter finalizers on EC2NodeClass + NodePool ==="
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
echo "=== Phase 3: Terminate any Karpenter-provisioned EC2 instances ==="
# Filter by presence of the karpenter.sh/nodepool tag key. Compound
# tag filters with `Values=*` don't behave like a wildcard match in
# the EC2 API — `tag-key` is the right filter shape for "instances
# carrying this tag at all". Cross-reference against the cluster-name
# tag so we only terminate instances belonging to this deployment
# (multi-cluster accounts may have other agent-sandbox-flavored
# clusters running alongside).
#
# Single scan is sufficient because Phase 1 stopped the controller.
KARPENTER_INSTANCES=$(aws ec2 describe-instances \
    --region "$REGION" \
    --filters "Name=tag-key,Values=karpenter.sh/nodepool" \
              "Name=tag:aws:eks:cluster-name,Values=$CLUSTER_NAME" \
              "Name=instance-state-name,Values=running,pending,stopping" \
    --query "Reservations[].Instances[].InstanceId" \
    --output text 2>/dev/null || echo "")

# Secondary sweep: if the primary filter returns empty, look up
# instances by Karpenter NodePool tag value. NodePool names follow
# the pattern <cluster-name>-* (e.g., agent-sandbox-gvisor) which is
# more reliable than the aws:eks:cluster-name tag — the latter has
# observed eventual-consistency lag that can leave instances
# transiently invisible to the primary filter, leading to orphaned
# nodes that block VPC delete in Phase 4.
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

    # Run base cleanup with retries. The cluster API gets unstable as
    # destroy progresses (provider auth tokens expire, EKS API drops
    # the cluster mid-flight), and the helm/kubernetes providers can
    # error out with "Unauthorized" or "context deadline exceeded".
    # Retry pattern: try base cleanup, check if VPC + cluster are
    # gone, retry with raw `terraform destroy` if either is still
    # present (state-driven destroy doesn't need cluster API).
    for attempt in 1 2 3; do
        echo ""
        echo "  Base destroy attempt $attempt..."
        # Subshell so the script's non-zero exits don't kill our set -e.
        ( cd "$LOCAL_DIR" && bash ./cleanup.sh ) || true

        # Check whether resources we expect destroyed are actually gone.
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
        # Raw destroy (no rayjob preamble) — state-driven, doesn't need
        # cluster API. Captures the chronic "Unauthorized"/helm-provider
        # tail failures that don't actually leave terraform state in a
        # corrupted shape.
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

# Cluster security groups — EKS auto-creates `eks-cluster-sg-<cluster>-<id>`
# and is responsible for deleting it on cluster delete. Stale ENI
# references can leave it dangling, blocking VPC destroy. If a VPC
# matching this deployment still exists, sweep its SGs (everything
# but `default`) and retry VPC delete via terraform.
LINGERING_VPC=$(aws ec2 describe-vpcs \
    --region "$REGION" \
    --filters "Name=tag:Name,Values=${CLUSTER_NAME}" \
    --query "Vpcs[].VpcId" \
    --output text 2>/dev/null || echo "")
if [ -n "$LINGERING_VPC" ]; then
    echo "  VPC $LINGERING_VPC still present after base destroy — sweeping cluster SGs..."
    SG_IDS=$(aws ec2 describe-security-groups \
        --region "$REGION" \
        --filters "Name=vpc-id,Values=$LINGERING_VPC" \
        --query "SecurityGroups[?GroupName!='default'].GroupId" \
        --output text 2>/dev/null || echo "")
    if [ -n "$SG_IDS" ]; then
        for sg in $SG_IDS; do
            echo "    Deleting security group $sg"
            aws ec2 delete-security-group --region "$REGION" --group-id "$sg" >/dev/null 2>&1 || true
        done
    fi
    if [ -d "$LOCAL_DIR" ]; then
        echo "  Retrying VPC destroy via terraform..."
        ( cd "$LOCAL_DIR" && terraform destroy -auto-approve -var-file=../blueprint.tfvars -target=module.vpc ) || true
    fi
fi

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
if [ "$PHASE_4_SUCCESS" = "true" ]; then
    echo "=== Cleanup complete ==="
    echo ""
    echo "Resources removed:"
    echo "  - All terraform-managed resources (EKS, VPC, IAM, KMS, log groups)"
    echo "  - Bedrock IRSA role (via the egress example's uninstall phase)"
    echo "  - Karpenter-provisioned EC2 instances"
    echo "  - Auxiliary AWS resources (placement groups, KMS aliases, log groups)"
else
    echo "=== Cleanup partially complete ==="
    echo ""
    echo "Phase 4 (terraform destroy) did not fully succeed after 3 retries."
    echo "Manual cleanup may be required for:"
    echo "  - VPC tagged Name=$CLUSTER_NAME"
    echo "  - EKS cluster $CLUSTER_NAME (if still present)"
    echo "  - Any orphaned ENIs / EC2 instances tagged with the cluster name"
    echo ""
    echo "To retry the base destroy manually:"
    echo "  cd $LOCAL_DIR"
    echo "  terraform destroy -auto-approve -var-file=../blueprint.tfvars"
    exit 1
fi
