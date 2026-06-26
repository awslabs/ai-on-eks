#!/usr/bin/env bash
# node-debug.sh — pre-staged on-node access for conformance debugging.
#
# Why this exists: during E2E validation, two things made ad-hoc node
# inspection slow and unreliable:
#   1. Fresh Karpenter nodes take a few minutes to register with SSM, so
#      `aws ssm send-command` fails with InvalidInstanceId right when you
#      want to look.
#   2. The `kube-system` namespace is PodSecurity-restricted, so privileged
#      hostPID debug pods are rejected there. The `default` namespace works.
#
# This runs a long-lived privileged hostPID pod in `default`, pinned to the
# target node, and execs a host-namespace command into it via nsenter — no
# SSM dependency, no registration wait. The pod sleeps so you can exec
# repeatedly without racing a one-shot container's completion.
#
# Usage:
#   ./node-debug.sh <node-name> [command...]
#   ./node-debug.sh ip-100-65-180-171.us-west-2.compute.internal
#   ./node-debug.sh <node> "findmnt --source soci"
#
# Default command bundles the SOCI + Kata+FC checks the E2E loop cares about.
#
# Cleanup:
#   ./node-debug.sh --clean        # delete all node-debug pods

set -euo pipefail

NS="default"

if [[ "${1:-}" == "--clean" ]]; then
  kubectl -n "$NS" delete pod -l app=node-debug --ignore-not-found
  exit 0
fi

NODE="${1:?usage: node-debug.sh <node-name> [command...]   (or --clean)}"
shift || true

# Default diagnostic bundle: active snapshotter, SOCI state, kata runtime.
DEFAULT_CMD='echo "== active snapshotter ==";
crictl info 2>/dev/null | grep -i snapshotter || true;
echo "== soci ==";
systemctl is-active soci-snapshotter 2>/dev/null || echo inactive;
findmnt --source soci 2>/dev/null || echo "no soci mounts";
echo "== kata-fc runtime ==";
crictl info 2>/dev/null | grep -i kata-fc || echo "kata-fc not configured";
echo "== devmapper ==";
dmsetup ls 2>/dev/null | grep -i thinpool || echo "no thinpool"'

CMD="${*:-$DEFAULT_CMD}"

# Pod name derived from the node (DNS-safe, length-capped).
POD="node-debug-$(echo "$NODE" | tr './' '--' | tr -cd 'a-z0-9-' | cut -c1-50)"

if ! kubectl -n "$NS" get pod "$POD" >/dev/null 2>&1; then
  kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  namespace: ${NS}
  labels:
    app: node-debug
spec:
  nodeName: ${NODE}
  hostPID: true
  hostNetwork: true
  restartPolicy: Never
  tolerations:
    - operator: Exists   # land on tainted tiers (gvisor, kata-fc) too
  containers:
    - name: debug
      image: public.ecr.aws/docker/library/busybox:latest
      securityContext:
        privileged: true
      command: ["sleep", "3600"]
EOF
  kubectl -n "$NS" wait --for=condition=Ready "pod/${POD}" --timeout=120s
fi

# nsenter into the host's mount/uts/ipc/net/pid namespaces (PID 1) and run.
kubectl -n "$NS" exec "$POD" -- nsenter -t 1 -m -u -i -n -p -- sh -c "$CMD"
