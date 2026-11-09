#!/bin/bash
set -u
OUT=/tmp/aiconf-runs/v1.36/dra
mkdir -p "$OUT"
cd /tmp/aiconf-test
export KUBECONFIG=~/.kube/config
kubectl config use-context eks-136-dra >/dev/null
go run gotest.tools/gotestsum@latest \
  --jsonfile "$OUT/results.json" --junitfile "$OUT/junit.xml" --format standard-verbose \
  -- -v -count=1 ./test -run TestSecureAcceleratorAccess -timeout 60m \
  > "$OUT/e2e.log" 2>&1
echo "exit=$?" | tee -a "$OUT/e2e.log"
grep -E '^(--- (PASS|FAIL)|ok |FAIL)|allocation mode' "$OUT/e2e.log"
