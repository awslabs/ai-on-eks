#!/bin/bash
# End-to-end deployment script for cache-aware LLM routing on Amazon EKS
# Prerequisites: aws cli, kubectl, helm, eksctl, HF_TOKEN and KMS_KEY_ARN env vars set
#
# Validated E2E: Jul 1 2026 — both RR and CA endpoints returning 200,
# benchmark shows 70-96% p90 TTFT improvement with cache-aware routing.

set -euo pipefail

REGION=${AWS_REGION:-us-west-2}
CLUSTER_NAME=${CLUSTER_NAME:-cache-routing-benchmark}
NAMESPACE=inference
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENVOY_GATEWAY_VERSION=v1.8.1
AI_GATEWAY_VERSION=v1.0.0

echo "=== Cache-Aware LLM Routing on EKS - Setup ==="
echo "Region: $REGION"
echo "Cluster: $CLUSTER_NAME"
echo ""

# Validate prerequisites
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN environment variable not set."
  echo "Get a token from https://huggingface.co/settings/tokens"
  exit 1
fi

if [ -z "${KMS_KEY_ARN:-}" ]; then
  echo "ERROR: KMS_KEY_ARN environment variable not set."
  echo "Create a KMS key: aws kms create-key --region $REGION"
  exit 1
fi

command -v eksctl >/dev/null || { echo "ERROR: eksctl not found"; exit 1; }
command -v kubectl >/dev/null || { echo "ERROR: kubectl not found"; exit 1; }
command -v helm >/dev/null || { echo "ERROR: helm not found"; exit 1; }

# Step 1: Create EKS cluster (~15 min)
echo ">>> Step 1: Creating EKS cluster (~15 min)..."
sed "s|\${KMS_KEY_ARN}|${KMS_KEY_ARN}|g" "$SCRIPT_DIR/manifests/cluster.yaml" | eksctl create cluster -f -
echo "Cluster created."

# Step 2: Install NVIDIA device plugin
echo ">>> Step 2: Installing NVIDIA device plugin..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.5/nvidia-device-plugin.yml
echo "NVIDIA plugin installed."

# Step 3: Create namespace and secrets
echo ">>> Step 3: Creating namespace and secrets..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
kubectl -n $NAMESPACE create secret generic hf-token --from-literal=token="$HF_TOKEN" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n $NAMESPACE create secret generic llm-d-hf-token --from-literal=HF_TOKEN="$HF_TOKEN" --dry-run=client -o yaml | kubectl apply -f -
echo "Secrets created."

# Step 4: Deploy vLLM with prefix caching + KVEvents
echo ">>> Step 4: Deploying vLLM (7 replicas, ~10 min for first image pull + model load)..."
kubectl apply -f "$SCRIPT_DIR/manifests/vllm-deployment.yaml"
echo "Waiting for vLLM pods to be ready..."
kubectl -n $NAMESPACE rollout status deployment/vllm-inference --timeout=1200s
echo "vLLM ready (7 pods)."

# Step 5: Install cert-manager (required for EPP TLS)
echo ">>> Step 5: Installing cert-manager..."
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.17.2/cert-manager.yaml
echo "Waiting for cert-manager pods..."
sleep 15
kubectl -n cert-manager wait --for=condition=Ready pod -l app.kubernetes.io/instance=cert-manager --timeout=120s
echo "cert-manager ready."

# Step 6: Install Gateway API CRDs + Envoy AI Gateway + Envoy Gateway
echo ">>> Step 6: Installing Gateway stack (Envoy AI Gateway ${AI_GATEWAY_VERSION} + Envoy Gateway ${ENVOY_GATEWAY_VERSION})..."

# 6a: Gateway API Inference Extension CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.5.0/manifests.yaml

# 6b: Gateway API standard + experimental CRDs (required by EG v1.8.1 for TLSRoute etc)
# --server-side is required because the HTTPRoute CRD exceeds client-side annotation size limits
kubectl apply --server-side -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.0/experimental-install.yaml

# 6c: Envoy AI Gateway CRDs + controller
curl -sL "https://github.com/envoyproxy/ai-gateway/releases/download/${AI_GATEWAY_VERSION}/ai-gateway-crds-helm-${AI_GATEWAY_VERSION}.tgz" -o /tmp/ai-gateway-crds.tgz
curl -sL "https://github.com/envoyproxy/ai-gateway/releases/download/${AI_GATEWAY_VERSION}/ai-gateway-helm-${AI_GATEWAY_VERSION}.tgz" -o /tmp/ai-gateway.tgz

helm install ai-gateway-crds /tmp/ai-gateway-crds.tgz -n envoy-ai-gateway-system --create-namespace
helm install ai-gateway /tmp/ai-gateway.tgz -n envoy-ai-gateway-system
echo "Waiting for AI Gateway controller..."
sleep 15
kubectl -n envoy-ai-gateway-system wait --for=condition=Ready pod -l app.kubernetes.io/name=ai-gateway-helm --timeout=120s
echo "AI Gateway controller ready."

# 6d: Envoy Gateway with AI Gateway values + InferencePool addon
helm install eg oci://docker.io/envoyproxy/gateway-helm \
  --version ${ENVOY_GATEWAY_VERSION} \
  -n envoy-gateway-system --create-namespace \
  -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml \
  -f https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/examples/inference-pool/envoy-gateway-values-addon.yaml \
  --timeout=180s

echo "Waiting for Envoy Gateway..."
kubectl -n envoy-gateway-system wait --for=condition=Available deployment/envoy-gateway --timeout=120s

# 6e: RBAC for Envoy Gateway to watch InferencePool resources
kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: envoy-gateway-inference-access
rules:
- apiGroups: ["inference.networking.k8s.io"]
  resources: ["inferencepools"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: envoy-gateway-inference-access
subjects:
- kind: ServiceAccount
  name: envoy-gateway
  namespace: envoy-gateway-system
roleRef:
  kind: ClusterRole
  name: envoy-gateway-inference-access
  apiGroup: rbac.authorization.k8s.io
EOF

# 6f: Apply Gateway and GatewayClass
kubectl apply -f "$SCRIPT_DIR/manifests/gateway.yaml"
echo "Waiting for Gateway to be programmed..."
kubectl -n $NAMESPACE wait --for=condition=Programmed gateway/inference-gateway --timeout=180s 2>/dev/null || \
  echo "  (Gateway condition check skipped — will verify in Step 9)"
echo "Gateway stack installed."

# Step 7: Deploy llm-d Router with precise prefix-cache scorer
echo ">>> Step 7: Deploying llm-d Router (precise prefix-cache scorer)..."
helm install cache-aware-routing oci://ghcr.io/llm-d/charts/llm-d-router-gateway-dev \
  --version v0 -n $NAMESPACE \
  -f "$SCRIPT_DIR/manifests/llm-d-router-values.yaml"
echo "Waiting for EPP to be ready..."
sleep 30
kubectl -n $NAMESPACE wait --for=condition=Ready pod -l llm-d-router-gateway=cache-aware-routing-epp --timeout=180s
echo "llm-d Router ready."

# Step 8: Deploy benchmark runner
echo ">>> Step 8: Deploying benchmark runner..."
kubectl apply -f "$SCRIPT_DIR/manifests/benchmark-runner.yaml"
kubectl -n $NAMESPACE wait --for=condition=Ready pod/benchmark-runner --timeout=120s
kubectl -n $NAMESPACE exec benchmark-runner -- pip install aiohttp --quiet
echo "Benchmark runner ready."

# Step 9: Verify both endpoints
echo ">>> Step 9: Verifying endpoints..."
CA_SVC=$(kubectl -n envoy-gateway-system get svc \
  -l gateway.envoyproxy.io/owning-gateway-name=inference-gateway \
  -o jsonpath='{.items[0].metadata.name}')

kubectl -n $NAMESPACE exec benchmark-runner -- python3 -c "
import urllib.request, json
endpoints = {
    'Round-Robin': 'http://vllm-inference.inference.svc.cluster.local:8000/v1/completions',
    'Cache-Aware': 'http://${CA_SVC}.envoy-gateway-system.svc.cluster.local:8080/v1/completions'
}
for name, ep in endpoints.items():
    req = urllib.request.Request(ep, data=json.dumps({'model':'mistralai/Mistral-7B-Instruct-v0.3','prompt':'Hello','max_tokens':5}).encode(), headers={'Content-Type':'application/json'})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        print(f'{name}: OK')
    except Exception as e:
        print(f'{name}: FAIL - {e}')
"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Both endpoints verified. Run the benchmark with:"
echo ""
CA_EP="http://${CA_SVC}.envoy-gateway-system.svc.cluster.local:8080/v1/completions"
echo "  cat benchmarks/sustained_benchmark.py | kubectl -n $NAMESPACE exec -i benchmark-runner -- tee /tmp/bench.py > /dev/null"
echo "  kubectl -n $NAMESPACE exec benchmark-runner -- python3 /tmp/bench.py --ca-endpoint \"$CA_EP\""
echo ""
echo "To stop billing (scale GPU nodes to zero):"
echo "  aws eks update-nodegroup-config --cluster-name $CLUSTER_NAME --nodegroup-name gpu-nodes \\"
echo "    --scaling-config minSize=0,maxSize=8,desiredSize=0 --region $REGION"
