#!/bin/bash
# Artifact set B: TestAcceleratorClusterAutoscaling, run on its own.
# It requires an accelerator pool containing no non-DaemonSet workloads, so the
# gpu-hold Deployment that keeps a GPU node alive for the other tests is removed
# first. Karpenter's WhenEmpty consolidation ignores DaemonSet pods, which is why
# the two cannot run together.
set -u
OUT=/tmp/aiconf-runs/v1.37/dra-setB
mkdir -p "$OUT"
cd /tmp/aiconf-test
export KUBECONFIG=~/.kube/config
kubectl config use-context eks-auto-cluster >/dev/null
go run gotest.tools/gotestsum@latest \
  --jsonfile "$OUT/results.json" --junitfile "$OUT/junit.xml" --format standard-verbose \
  -- -v -count=1 ./test -run TestAcceleratorClusterAutoscaling -timeout 90m -args \
  -autoscaler-node-pool-label=karpenter.sh/nodepool=gpu-dra \
  -autoscaler-scale-up-timeout=20m -autoscaler-scale-down-timeout=25m \
  > "$OUT/e2e.log" 2>&1
echo "exit=$?" | tee -a "$OUT/e2e.log"
grep -E '^(--- (PASS|FAIL)|ok |FAIL)' "$OUT/e2e.log"
