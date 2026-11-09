## Purpose

Automated test evidence for the Amazon EKS submission to [Kubernetes AI Conformance](https://github.com/cncf/k8s-ai-conformance) for Kubernetes 1.36, produced with the upstream suite in [kubernetes-sigs/ai-conformance](https://github.com/kubernetes-sigs/ai-conformance/tree/main/test).

## Cluster under test

| | |
|---|---|
| Kubernetes version | `1.36` |
| Server version | `v1.36.5-eks-577a51e` |
| EKS platform version | `eks.17` |
| Cluster type | EKS Auto Mode |
| Accelerator node | `g6e.4xlarge`, one NVIDIA L40S |
| Allocation mechanism | **DRA** — NVIDIA DRA driver, `gpu.nvidia.com` DeviceClass |

EKS Auto Mode includes the NVIDIA DRA driver in the node runtime, so no driver, device plugin or GPU operator is installed. The only object created is the `DeviceClass`.

## Result

`TestSecureAcceleratorAccess` passes. `junit.xml` reports 4 test cases, 0 failures, 0 errors.

```
Auto-detected allocation mode: dra
--- PASS: TestSecureAcceleratorAccess
    --- PASS: TestSecureAcceleratorAccess/PositiveAccessTest
    --- PASS: TestSecureAcceleratorAccess/NegativeIsolationTest
    --- PASS: TestSecureAcceleratorAccess/MultiContainerIsolationTest
```

## Reproducing

```
kubectl apply -f deviceclass-nvidia.yaml
kubectl apply -f nodepool-gpu-dra.yaml
./run.sh <kube-context>
```

| File | Contents |
|---|---|
| [e2e.log](./e2e.log) | Full test execution log |
| [junit.xml](./junit.xml) | JUnit XML report |
| [results.json](./results.json) | `go test -json` output |
| [deviceclass-nvidia.yaml](./deviceclass-nvidia.yaml) | `gpu.nvidia.com` DeviceClass |
| [nodepool-gpu-dra.yaml](./nodepool-gpu-dra.yaml) | EKS Auto Mode GPU NodePool in DRA mode |
| [run.sh](./run.sh) | Test invocation that produced the artifacts |
