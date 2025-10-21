# EKS K8s Conformance: HPA Custom Metrics

A repository to demonstrate EKS Kubernetes conformance for HPA Custom Metrics using GPU Metrics

## Steps

### Set up the Environment

```bash
eksctl create cluster -f gpu-cluster.yaml --install-nvidia-plugin=false
```

### Install Kube Prometheus Stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts && \
helm repo update

helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
 --values kube-prometheus-stack-values.yaml
```


### Install GPU Operator

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && \
helm repo update

helm upgrade --install gpu-operator nvidia/gpu-operator \
 --namespace nvidia \
 --create-namespace \
 --version v25.3.4 \
 --values gpu-operator-values.yaml
```

### Create Prometheus Rule
```bash
kubectl apply -f cuda-prometheusrule.yaml
```

### Install Prometheus Adapter

The Prometheus adapter is used to enable querying `custom.metrics.k8s.io` for Prometheus metrics

```bash
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --set prometheus.url="http://kube-prometheus-stack-prometheus.default.svc.cluster.local"
```

### Test Kubernetes Custom Metrics Endpoint

```bash
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq -r . | grep cuda_gpu
```
Should present cuda_gpu in the results

### Create a Deployment for scaling

```bash
kubectl apply -f cuda-deployment.yaml
```

### Create HPA to Scale deployment based on GPU Utilization

```bash
kubectl apply -f cuda-hpa.yaml
```

### Simulate load on the GPU

```bash

kubectl exec -it deployment/cuda -- bash

for (( c=1; c<=5000; c++ )); do ./vectorAdd; done & \
for (( c=1; c<=5000; c++ )); do ./vectorAdd; done & \
for (( c=1; c<=5000; c++ )); do ./vectorAdd; done & \
for (( c=1; c<=5000; c++ )); do ./vectorAdd; done & \
for (( c=1; c<=5000; c++ )); do ./vectorAdd; done & \
for (( c=1; c<=5000; c++ )); do ./vectorAdd; done & \
for (( c=1; c<=5000; c++ )); do ./vectorAdd; done & \
for (( c=1; c<=5000; c++ )); do ./vectorAdd; done &
```

With the added GPU load, the deployment scale up to the 3 replica maximum
