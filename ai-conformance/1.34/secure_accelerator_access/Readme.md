## Purpose
The purpose of this document is to show how Amazon EKS meets the [Secure Accelerator Access requirement](https://github.com/cncf/ai-conformance/blob/main/docs/AIConformance-1.34.yaml#L65) for [Kubernetes
AI Conformance](https://github.com/cncf/ai-conformance/tree/main).

## Requirement
Ensure that access to accelerators from within containers is properly isolated and mediated by the
Kubernetes resource management framework (device plugin or DRA) and container runtime, preventing unauthorized access
or interference between workloads.

## How to test
- [Test 1](./EKS-Conformance-Test-1.md): Deploy a Pod to a node with available accelerators, without requesting accelerator resources in the Pod spec.
Execute a command in the Pod to probe for accelerator devices, and the command should fail or report that no accelerator
devices are found.
- [Test 2](./EKS-Conformance-Test-2.md): Create two Pods, each is allocated an accelerator resource. Execute a command in one Pod to attempt to access
the other Pod’s accelerator, and should be denied.

## EKS Cluster Setup

**Step 1**: Create an EKS cluster with at least 1 node that has multiple GPUs. ([gpu-cluster.yaml](./gpu-cluster.yaml))

```
eksctl create cluster -f gpu-cluster.yaml --install-nvidia-plugin=false
```

**Step 2**: Install the NVIDIA GPU operator

```
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
    && helm repo update
```
See ([gpu-operator-values.yaml](./gpu-operator-values.yaml))
```
helm install gpu-operator nvidia/gpu-operator \
 --namespace nvidia \
 --create-namespace \
 --version v25.3.4 \
 --values gpu-operator-values.yaml
```

**Step 3**: Install NVIDIA DRA driver

See ([gpu-dra-values.yaml](./gpu-dra-values.yaml))
```
helm install nvidia-dra-driver-gpu nvidia/nvidia-dra-driver-gpu \
 --version 25.3.2 \
 --namespace nvidia \
 -f gpu-dra-values.yaml
```
