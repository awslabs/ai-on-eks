#!/bin/bash
# AI on EKS — shared conformance driver.
#
# Runs a component-supplied declarative spec (conformance.yaml):
#   apply manifests -> wait readiness -> run assertions -> report -> cleanup.
#
# Components author a spec, not a script. See ./README.md for the schema and
# the supported assertion types. Seeds issue #334 (cross-component reuse).
#
# Usage:
#   run-conformance.sh <path-to-conformance.yaml>
#
# Exit 0 on all-pass, 1 on any failure. No interactive prompts.
# Requires: kubectl, aws CLI v2, jq, yq.
#
# NOTE: candidate harness — like every blueprint here it is only "done" once it
# clears end-to-end on a live cluster (spin-up -> conformance -> cleanup) with
# zero manual intervention.

set -euo pipefail

SPEC="${1:-}"
[ -n "$SPEC" ] && [ -f "$SPEC" ] || { echo "usage: $0 <conformance.yaml>" >&2; exit 2; }
SPEC_DIR="$(cd "$(dirname "$SPEC")" && pwd)"

for tool in kubectl aws jq yq; do
    command -v "$tool" >/dev/null 2>&1 || { echo "FATAL: required tool '$tool' not found" >&2; exit 2; }
done

log()  { echo "[$(date +%H:%M:%S)] $*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

# ---- spec accessors (yq) ----------------------------------------------------
yq_get() { yq -r "$1" "$SPEC"; }

NAME="$(yq_get '.metadata.name')"
NS="$(yq_get '.spec.namespace // "default"')"
CLUSTER_NAME="${CLUSTER_NAME:-agent-sandbox}"

# ---- region resolution (matches infra/agent-sandbox cleanup + conformance) --
# Precedence: AWS_REGION > AWS_DEFAULT_REGION > kubectl context > us-west-2.
if [ -n "${AWS_REGION:-}" ]; then
    REGION="$AWS_REGION"
elif [ -n "${AWS_DEFAULT_REGION:-}" ]; then
    REGION="$AWS_DEFAULT_REGION"
else
    REGION="$(kubectl config current-context 2>/dev/null | awk -F':' '{print $4}' || echo "")"
    REGION="${REGION:-us-west-2}"
fi
log "Conformance spec='$NAME' cluster='$CLUSTER_NAME' region='$REGION' namespace='$NS'"

# ---- namespace -------------------------------------------------------------
# Ensure the target namespace exists. Running a spec standalone (without the
# full component install) shouldn't fail just because the namespace isn't
# there yet. Idempotent; not deleted on cleanup (cheap, and teardown removes
# the cluster anyway — avoids clobbering a namespace the component owns).
ensure_namespace() {
    [ "$NS" = "default" ] && return 0
    if ! kubectl get namespace "$NS" >/dev/null 2>&1; then
        log "Creating namespace $NS"
        kubectl create namespace "$NS" >/dev/null
    fi
}

# ---- manifest apply / cleanup ----------------------------------------------
APPLIED=()
apply_manifests() {
    local count m path
    count="$(yq_get '.spec.manifests | length')"
    for ((i=0; i<count; i++)); do
        m="$(yq_get ".spec.manifests[$i]")"
        path="$SPEC_DIR/$m"
        [ -f "$path" ] || fail "manifest not found: $path"
        log "Applying $m"
        kubectl apply -f "$path" >/dev/null
        APPLIED+=("$path")
    done
}

cleanup() {
    if [ "$(yq_get '.spec.cleanup.deleteManifests // "true"')" = "true" ]; then
        # reverse order
        for ((idx=${#APPLIED[@]}-1; idx>=0; idx--)); do
            log "Deleting ${APPLIED[$idx]##*/}"
            kubectl delete -f "${APPLIED[$idx]}" --ignore-not-found >/dev/null 2>&1 || true
        done
    fi
}
trap cleanup EXIT

# ---- readiness gates --------------------------------------------------------
wait_readiness() {
    local count kind name timeout
    count="$(yq_get '.spec.readiness | length')"
    for ((i=0; i<count; i++)); do
        kind="$(yq_get ".spec.readiness[$i].kind")"
        name="$(yq_get ".spec.readiness[$i].name")"
        timeout="$(yq_get ".spec.readiness[$i].timeout // \"300s\"")"
        log "Waiting for $kind/$name Ready (timeout $timeout)..."
        case "$kind" in
            Pod)        kubectl -n "$NS" wait --for=condition=Ready "pod/$name" --timeout="$timeout" >/dev/null \
                            || { kubectl -n "$NS" describe "pod/$name" >&2; fail "$kind/$name not Ready"; } ;;
            Deployment) kubectl -n "$NS" rollout status "deployment/$name" --timeout="$timeout" >/dev/null \
                            || fail "$kind/$name not rolled out" ;;
            *)          fail "unsupported readiness kind: $kind" ;;
        esac
    done
}

# ---- assertion helpers ------------------------------------------------------
match_value() { # $1=actual $2=mode $3=expected
    case "$2" in
        contains) [[ "$1" == *"$3"* ]] ;;
        equals)   [[ "$1" == "$3" ]] ;;
        regex)    [[ "$1" =~ $3 ]] ;;
        *)        return 2 ;;
    esac
}

node_for_pod() { kubectl -n "$NS" get "pod/$1" -o jsonpath='{.spec.nodeName}'; }

instance_for_node() {
    aws ec2 describe-instances --region "$REGION" \
        --filters "Name=private-dns-name,Values=$1" \
        --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null
}

ssm_run() { # $1=instance-id $2=command -> stdout
    local cid
    cid="$(aws ssm send-command --region "$REGION" --instance-ids "$1" \
        --document-name "AWS-RunShellScript" --parameters commands="[\"$2\"]" \
        --query "Command.CommandId" --output text)"
    aws ssm wait command-executed --region "$REGION" --command-id "$cid" --instance-id "$1" 2>/dev/null || true
    aws ssm list-command-invocations --region "$REGION" --command-id "$cid" --instance-id "$1" \
        --details --query "CommandInvocations[0].CommandPlugins[0].Output" --output text
}

# ---- assertion dispatch -----------------------------------------------------
run_assertion() { # $1=index
    local i="$1" name type desc
    name="$(yq_get ".spec.assertions[$i].name")"
    type="$(yq_get ".spec.assertions[$i].type")"
    desc="$(yq_get ".spec.assertions[$i].description // \"\"")"
    log "  [$name] $desc"

    case "$type" in
        podReady)
            local target
            target="$(yq_get ".spec.assertions[$i].target")"
            kubectl -n "$NS" wait --for=condition=Ready "pod/$target" --timeout=60s >/dev/null \
                || fail "[$name] pod/$target not Ready"
            ;;
        podExec)
            local target container cmd mode expected out
            target="$(yq_get ".spec.assertions[$i].target")"
            container="$(yq_get ".spec.assertions[$i].container // \"\"")"
            cmd="$(yq_get ".spec.assertions[$i].command")"
            mode="$(yq_get ".spec.assertions[$i].match")"
            expected="$(yq_get ".spec.assertions[$i].expected")"
            if [ -n "$container" ]; then
                out="$(kubectl -n "$NS" exec "pod/$target" -c "$container" -- sh -c "$cmd" 2>&1 || true)"
            else
                out="$(kubectl -n "$NS" exec "pod/$target" -- sh -c "$cmd" 2>&1 || true)"
            fi
            match_value "$out" "$mode" "$expected" || fail "[$name] exec output did not $mode '$expected' (got: $out)"
            ;;
        nodeExec)
            local podref cmd mode expected node iid out
            podref="$(yq_get ".spec.assertions[$i].podRef")"
            cmd="$(yq_get ".spec.assertions[$i].command")"
            mode="$(yq_get ".spec.assertions[$i].match")"
            expected="$(yq_get ".spec.assertions[$i].expected")"
            node="$(node_for_pod "$podref")"; [ -n "$node" ] || fail "[$name] no node for pod/$podref"
            iid="$(instance_for_node "$node")"; [ -n "$iid" ] && [ "$iid" != "None" ] || fail "[$name] no instance for node $node"
            out="$(ssm_run "$iid" "$cmd")"
            match_value "$out" "$mode" "$expected" || fail "[$name] node command did not $mode '$expected' (got: $out)"
            ;;
        jsonpath)
            local resource path expected actual
            resource="$(yq_get ".spec.assertions[$i].resource")"
            path="$(yq_get ".spec.assertions[$i].path")"
            expected="$(yq_get ".spec.assertions[$i].expected")"
            actual="$(kubectl -n "$NS" get "$resource" -o jsonpath="$path" 2>/dev/null || echo "")"
            [ "$actual" = "$expected" ] || fail "[$name] jsonpath '$path' = '$actual', expected '$expected'"
            ;;
        *)
            fail "[$name] unsupported assertion type: $type"
            ;;
    esac
    log "  [$name] PASS"
}

main() {
    ensure_namespace
    apply_manifests
    wait_readiness
    local count
    count="$(yq_get '.spec.assertions | length')"
    log "Running $count assertion(s)..."
    for ((i=0; i<count; i++)); do run_assertion "$i"; done
    log ""
    log "PASS: conformance spec '$NAME' succeeded ($count assertions)."
}

main "$@"
