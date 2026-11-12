#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# In-cluster image build with Kaniko — NO local Docker required. VLLM-OPENAI variant.
# Builds ./Dockerfile (vllm/vllm-openai + Ray + Run:AI streamer) on an existing EKS node and
# pushes to your ECR. Run from this directory (image/).
#
# Produces an image, e.g. <ACCOUNT>.dkr.ecr.us-east-2.amazonaws.com/vllm-405b:v0.26.0-runai
#   -> import OK: vllm 0.26.0 | torch 2.11.0+cu130 | ray 2.59.0  -> served 405B (/health 200).
#
# Prereqs: kubectl context on the target EKS cluster, `aws` creds (export AWS_PROFILE if needed), an
# ECR repo you can push to, and a CPU (non-GPU) node present. Base = vllm/vllm-openai on Docker Hub
# (no pull auth; if Docker Hub rate-limits Kaniko, add a docker.io cred to the config.json secret).
# ---------------------------------------------------------------------------
set -euo pipefail

REGION="${REGION:-us-east-2}"
ACCOUNT_ID="${ACCOUNT_ID:?set ACCOUNT_ID=<your 12-digit account id>}"
ECR_REPO="${ECR_REPO:-vllm-405b}"
IMAGE_TAG="${IMAGE_TAG:-v0.26.0-runai}"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# vllm-openai base (Docker Hub). Override to pin a different vLLM version.
BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.26.0}"   # validated pin; newer tags on Docker Hub — override with BASE_IMAGE

# Any Linux/x86 CPU node (no GPU is used by the build). Override to match an instance type in your cluster.
BUILD_INSTANCE_TYPE="${BUILD_INSTANCE_TYPE:-c6a.xlarge}"

echo "Building ${IMAGE_URI}"
echo "  from base ${BASE_IMAGE}"
aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" >/dev/null 2>&1 || true

# 1) ECR push credential as a docker config.json secret (token ~12h; re-run if it expires mid-build)
TOKEN="$(aws ecr get-login-password --region "$REGION")"
AUTH="$(printf 'AWS:%s' "$TOKEN" | base64 | tr -d '\n')"
TMP="$(mktemp)"
printf '{"auths":{"%s.dkr.ecr.%s.amazonaws.com":{"auth":"%s"}}}' "$ACCOUNT_ID" "$REGION" "$AUTH" > "$TMP"
kubectl delete secret kaniko-ecr --ignore-not-found >/dev/null
kubectl create secret generic kaniko-ecr --from-file=config.json="$TMP"
rm -f "$TMP"

# 2) Dockerfile as the Kaniko build context (configmap)
kubectl delete configmap kaniko-dockerfile-vllm --ignore-not-found >/dev/null
kubectl create configmap kaniko-dockerfile-vllm --from-file=Dockerfile=./Dockerfile

# 3) Kaniko build+push Job
kubectl delete job kaniko-vllmopenai-build --ignore-not-found --wait=true >/dev/null 2>&1 || true
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: kaniko-vllmopenai-build
  namespace: default
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        node.kubernetes.io/instance-type: ${BUILD_INSTANCE_TYPE}
      containers:
        - name: kaniko
          image: gcr.io/kaniko-project/executor:v1.23.2
          args:
            - "--context=dir:///workspace"
            - "--dockerfile=/workspace/Dockerfile"
            - "--destination=${IMAGE_URI}"
            - "--build-arg=BASE_IMAGE=${BASE_IMAGE}"
            - "--cache=false"
          resources:
            requests: { cpu: "2", memory: "6Gi", ephemeral-storage: "40Gi" }
            limits:   { ephemeral-storage: "80Gi" }
          volumeMounts:
            - { name: dockerfile,    mountPath: /workspace }
            - { name: docker-config, mountPath: /kaniko/.docker }
      volumes:
        - name: dockerfile
          configMap: { name: kaniko-dockerfile-vllm }
        - name: docker-config
          secret:
            secretName: kaniko-ecr
            items: [ { key: config.json, path: config.json } ]
EOF

echo "Waiting for build to complete (~5-8 min)..."
kubectl wait --for=condition=complete job/kaniko-vllmopenai-build --timeout=1200s
kubectl logs -l job-name=kaniko-vllmopenai-build --tail=5

kubectl delete job kaniko-vllmopenai-build --ignore-not-found >/dev/null 2>&1 || true
kubectl delete configmap kaniko-dockerfile-vllm --ignore-not-found >/dev/null 2>&1 || true
kubectl delete secret kaniko-ecr --ignore-not-found >/dev/null 2>&1 || true

echo ""
echo "Pushed: ${IMAGE_URI}"
echo "Now:    terraform apply ... -var vllm_image=${IMAGE_URI}"
