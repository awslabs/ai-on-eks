#!/bin/bash
# Artifact set A: every 1.37 test except TestAcceleratorClusterAutoscaling and
# TestWorkloadSandboxing.
#   - autoscaling is split out because it requires an accelerator pool with no
#     non-DaemonSet workloads, while every other GPU test needs a node that
#     survives; Karpenter's WhenEmpty consolidation ignores DaemonSet pods, so
#     the two requirements cannot be satisfied in one run.
#   - workload_sandboxing is excluded: the upstream probe has a false negative on
#     microVM sandboxes (separate fix PR); that requirement is submitted with
#     manual attestation.
set -u
OUT=/tmp/aiconf-runs/v1.37/dra-setA
mkdir -p "$OUT"
cd /tmp/aiconf-test
export KUBECONFIG=~/.kube/config
kubectl config use-context eks-auto-cluster >/dev/null
RUN='TestSecureAcceleratorAccess|TestAcceleratorDriverRuntimeManagement|TestGangScheduling|TestHPAAutoscaling|TestAcceleratorPerformanceMetrics|TestAIServiceMetrics|TestGatewayAPIInference|TestRobustCRDControllerOperation'
go run gotest.tools/gotestsum@latest \
  --jsonfile "$OUT/results.json" --junitfile "$OUT/junit.xml" --format standard-verbose \
  -- -v -count=1 ./test -run "$RUN" -timeout 120m -args \
  -gang-scheduler-name=kueue -gang-scheduler-namespace=gang-conformance \
  -gang-job-labels=kueue.x-k8s.io/queue-name=conformance-lq \
  -hpa-custom-metric-name=ai_conformance_hpa_queue_depth -hpa-max-replicas=2 \
  -hpa-scale-up-timeout=20m -hpa-scale-down-timeout=15m \
  -accelerator-metrics-namespace=dcgm -accelerator-metrics-service-name=dcgm-exporter -accelerator-metrics-service-port=9400 \
  -service-metrics-prometheus-service=monitoring/prometheus-operated:9090 \
  -gateway-api-class=agentgateway-clusterip -gateway-api-timeout=12m \
  -operator=kubeflow-trainer -operator-runtime-name=torch-distributed \
  > "$OUT/e2e.log" 2>&1
echo "exit=$?" | tee -a "$OUT/e2e.log"
grep -E '^(--- (PASS|FAIL)|ok |FAIL)' "$OUT/e2e.log"
