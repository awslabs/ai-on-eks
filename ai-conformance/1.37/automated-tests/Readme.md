## Purpose

Automated test evidence for the Amazon EKS submission to [Kubernetes AI Conformance](https://github.com/cncf/k8s-ai-conformance) for Kubernetes 1.37, produced with the upstream suite in [kubernetes-sigs/ai-conformance](https://github.com/kubernetes-sigs/ai-conformance/tree/main/test).

## Cluster under test

| | |
|---|---|
| Kubernetes version | `1.37` |
| Server version | `v1.37.1-eks-577a51e` |
| EKS platform version | `eks.7` |
| Cluster type | EKS Auto Mode |
| Accelerator node | `g5.xlarge`, one NVIDIA A10G |
| Allocation mechanism | **DRA** — NVIDIA DRA driver, `gpu.nvidia.com` DeviceClass |

EKS Auto Mode includes the NVIDIA DRA driver in the node runtime, so no driver, device plugin or GPU operator is installed. The only object created is the `DeviceClass`.

## Results

Nine of the ten requirements with an automated test pass. `workload_sandboxing` is excluded and submitted with manual attestation; see [Excluded test](#excluded-test).

| Requirement | Test | Result |
|---|---|---|
| `secure_accelerator_access` | `TestSecureAcceleratorAccess` | PASS |
| `driver_runtime_management` | `TestAcceleratorDriverRuntimeManagement` | PASS |
| `ai_inference` | `TestGatewayAPIInference` | PASS |
| `gang_scheduling` | `TestGangScheduling` | PASS |
| `pod_autoscaling` | `TestHPAAutoscaling` | PASS |
| `accelerator_metrics` | `TestAcceleratorPerformanceMetrics` | PASS |
| `ai_service_metrics` | `TestAIServiceMetrics` | PASS |
| `robust_controller` | `TestRobustCRDControllerOperation` | PASS |
| `cluster_autoscaling` | `TestAcceleratorClusterAutoscaling` | PASS |

`junit.xml` reports 24 test cases with 0 failures and 0 errors; `cluster-autoscaling/junit.xml` reports 1 with 0 failures.


| Set | Files | Script |
|---|---|---|
| A — eight tests | `e2e.log`, `junit.xml`, `results.json` | [run-set-a.sh](./run-set-a.sh) |
| B — cluster autoscaling | `cluster-autoscaling/{e2e.log,junit.xml,results.json}` | [run-set-b.sh](./run-set-b.sh) |

Set B lives in a subdirectory rather than carrying a filename suffix because the conformance validator classifies an artifact purely by basename (`junit.xml`, `results.json`, `e2e.log`); a suffixed file is silently ignored.

Before running set B, let the pool settle to exactly one node and stay there. The test captures its pool baseline at the start, and if Karpenter is still converging, the baseline Pod can land on a node outside the captured set, which the test reports as `ENVIRONMENT ERROR: baseline Pod ... ran on node ... outside the captured pool`. The NodePool uses `consolidateAfter: 5m` for the same reason: at `1m` there is no usable window between freeing the pool and the test capturing its baseline, while `5m` still lets the test observe scale-down inside its 25m timeout.

## Components installed to exercise the requirements

These requirements ask whether the platform can install and run such a solution, so installing it is the test. None of these ship with EKS.

| Component | Version | Requirement | Config |
|---|---|---|---|
| Kueue | 0.14.3 | `gang_scheduling` | [kueue-queues.yaml](./kueue-queues.yaml) |
| NVIDIA DCGM exporter | chart default | `accelerator_metrics` | — |
| kube-prometheus-stack | chart default | `ai_service_metrics`, `pod_autoscaling` | — |
| prometheus-adapter | 5.3.0 | `pod_autoscaling` | [prometheus-adapter-values.yaml](./prometheus-adapter-values.yaml) |
| Kubeflow Trainer | v2.4.0 | `robust_controller` | — |
| agentgateway | v1.7.0 | `ai_inference` | ClusterIP GatewayClass, as in the suite's own reference setup |


## Reproducing

```
kubectl apply -f deviceclass-nvidia.yaml
kubectl apply -f nodepool-gpu-dra.yaml
kubectl apply -f kueue-queues.yaml
# install the components in the table above
./run-set-a.sh          # eight tests
./run-set-b.sh          # cluster autoscaling, after scaling gpu-hold to 0
```


## Excluded test

`TestWorkloadSandboxing` is not included. The probe fails any interface whose name matches `dummy*`, and the Linux dummy driver creates an unused `dummy0` at boot, so a correctly isolated sandbox is reported as `NET_ISOLATION=FAIL`. The interface is guest-local: on the same node the host namespace has no dummy interface at all, and with [network isolation](https://docs.aws.amazon.com/eks/latest/userguide/sandboxing-network-isolation.html) enabled the guest's `dummy0` carries no address, so the name is not evidence of host network visibility. With that pattern removed the test passes all four subtests against `isolated-sandbox`. A fix has been sent upstream; until it lands, `workload_sandboxing` is submitted with manual attestation against the published documentation.
