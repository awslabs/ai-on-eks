#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# In-cluster image build with Kaniko — NO local Docker required.
# Builds ./Dockerfile on an existing EKS node and pushes to your ECR.
# Run from this directory (the image/ dir).
#
# On build, the Dockerfile imports vllm, torch and ray and prints their versions
# (exact versions depend on the DLC base tag you build from).
#
# Prereqs: kubectl context on the target EKS cluster, `aws` creds, an ECR
# repo you can push to. The base DLC defaults to the PUBLIC ECR DLC (no pull
# auth needed) which is why Kaniko can pull it without extra credentials.
# ---------------------------------------------------------------------------
set -euo pipefail

REGION="${REGION:-us-west-2}"
ACCOUNT_ID="${ACCOUNT_ID:?set ACCOUNT_ID=<your 12-digit account id>}"
ECR_REPO="${ECR_REPO:-vllm-ray-runai}"
IMAGE_TAG="${IMAGE_TAG:-vllm-ray-runai}"
IMAGE_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}"

# Base DLC. Default = PUBLIC ECR DLC (no login needed), the rolling `server-cuda` tag per the gallery
# (https://aws.github.io/deep-learning-containers/vllm/). `server-cuda` is the current/patched tag; it
# is rolling (not version-pinned), so the image can change over time — pin to a specific digest if you
# need reproducibility. For the private/canonical DLC, see the gallery "Image Access" page for the URI
# and add that account's pull credential to the config.json secret below.
DLC_BASE="${DLC_BASE:-public.ecr.aws/deep-learning-containers/vllm:server-cuda}"

# Any Linux/x86 node in the cluster works (no GPU is used by the build). Default to a cheap CPU
# type — NOT a GPU node — so an image build never burns a p5 / Capacity-Block slot. Override
# BUILD_INSTANCE_TYPE to match an instance type present in your cluster if m6i.xlarge is absent.
BUILD_INSTANCE_TYPE="${BUILD_INSTANCE_TYPE:-m6i.xlarge}"

echo "Building ${IMAGE_URI}"
echo "  from base ${DLC_BASE}"
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
kubectl delete configmap kaniko-dockerfile --ignore-not-found >/dev/null
kubectl create configmap kaniko-dockerfile --from-file=Dockerfile=./Dockerfile

# 3) Kaniko build+push Job
kubectl delete job kaniko-vllm-build --ignore-not-found --wait=true >/dev/null 2>&1 || true
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: kaniko-vllm-build
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
            - "--build-arg=DLC_BASE=${DLC_BASE}"
            - "--cache=false"
          volumeMounts:
            - { name: dockerfile,    mountPath: /workspace }
            - { name: docker-config, mountPath: /kaniko/.docker }
      volumes:
        - name: dockerfile
          configMap: { name: kaniko-dockerfile }
        - name: docker-config
          secret:
            secretName: kaniko-ecr
            items: [ { key: config.json, path: config.json } ]
EOF

echo "Waiting for build to complete (~5-8 min)..."
kubectl wait --for=condition=complete job/kaniko-vllm-build --timeout=1200s
kubectl logs -l job-name=kaniko-vllm-build --tail=5

# cleanup build artifacts (optional)
kubectl delete job kaniko-vllm-build --ignore-not-found >/dev/null 2>&1 || true
kubectl delete configmap kaniko-dockerfile --ignore-not-found >/dev/null 2>&1 || true
kubectl delete secret kaniko-ecr --ignore-not-found >/dev/null 2>&1 || true

echo ""
echo "Pushed: ${IMAGE_URI}"
echo "Now:    terraform apply ... -var vllm_image=${IMAGE_URI}"
