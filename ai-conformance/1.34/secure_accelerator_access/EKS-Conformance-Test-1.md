### Secure Accelerator Access - Test 1

Deploy a Pod to a node with available accelerators, without requesting accelerator resources in the Pod spec.
Execute a command in the Pod to probe for accelerator devices, and the command should fail or report that no accelerator
devices are found.

**Step 1**: Create a ResourceClaimTemplate for a single GPU resource ([gpuResourceClaimTemplate.yaml](./gpuResourceClaimTemplate.yaml))

```
kubectl apply -f gpuResourceClaimTemplate.yaml
```

**Step 2**: Create a Deployment that requests the single GPU resource ([gpuDeploymentSinglePod.yaml](./gpuDeploymentSinglePod.yaml))

```
kubectl apply -f gpuDeploymentSinglePod.yaml
```

**Step 3**: Confirm pod can access GPU and resource claims allocated
```
kubectl logs -n gpu-test1  gpu-deployment-78df9479c4-dw6nb
GPU 0: NVIDIA A10G (UUID: GPU-490bc72f-1c3c-c960-4fe2-d42c7d90a1e1)
```
```
kubectl get resourceclaims -A
NAMESPACE   NAME                                         STATE                AGE
gpu-test1   gpu-deployment-78df9479c4-dw6nb-gpu0-6p5bf   allocated,reserved   33s
```
See [gpu-deployment-78df9479c4-dw6nb-gpu0-6p5bf](./gpu-deployment-78df9479c4-dw6nb-gpu0-6p5bf) file for output
```
kubectl get resourceclaim -n gpu-test1 gpu-deployment-78df9479c4-dw6nb-gpu0-6p5bf -o yaml
```

**Step 4**: Delete the deployment

```
kubectl -n gpu-test1 delete deployment gpu-deployment
```

**Step 5**: Remove the resource claim from the Deployment
Comment out the `spec.containers.resources` from [gpuDeploymentSinglePod.yaml](./gpuDeploymentSinglePod.yaml)


```
...
    spec:
      containers:
      - name: ctr0
        image: ubuntu:22.04
        command: ["bash", "-c"]
        args: ["nvidia-smi -L; trap 'exit 0' TERM; sleep 9999 & wait"]
        #resources:
        #  claims:
        #  - name: gpu0
      resourceClaims:
      - name: gpu0
        resourceClaimTemplateName: single-gpu
...
```

**Step 6**: Apply the new Deployment without resource claim

```
kubectl apply -f gpuDeploymentSinglePod.yaml
```

**Step 7**: Confirm Pod cannot access GPU when request for resource is removed

```
kubectl logs -n gpu-test1 gpu-deployment-7497756fc-nb25q
bash: line 1: nvidia-smi: command not found
```
