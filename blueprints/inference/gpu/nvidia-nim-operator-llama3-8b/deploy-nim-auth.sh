#!/bin/bash
# ------------------------------------------------------------------------------
# Script to Deploy NGC Authentication Secrets for NVIDIA Inference Microservice
# ------------------------------------------------------------------------------
# This script creates two Kubernetes secrets required for authenticating with NGC:
# 1. `ngc-secret` – Docker config secret to pull models from nvcr.io.
# 2. `ngc-api-secret` – Opaque secret containing the NGC API key for model access.
#
# Usage:
#   NGC_API_KEY="your-real-ngc-key" ./deploy-nim-auth.sh
#
# Notes:
# - Ensures the `nim-service` namespace exists.
# - Encodes secrets securely using jq and base64.
# ------------------------------------------------------------------------------


set -e

# Step 1: Export your NGC API key
# You can also set this in your shell before running the script
export NGC_API_KEY="${NGC_API_KEY:-<your-ngc-api-key>}"

if [ -z "$NGC_API_KEY" ]; then
  echo "[ERROR] NGC_API_KEY is not set. Please export it or pass it inline."
  exit 1
fi

# Step 2: Create namespace if it doesn't exist
kubectl get namespace nim-service >/dev/null 2>&1 || kubectl create namespace nim-service

# Step 3: Generate base64 encoded .dockerconfigjson
DOCKER_CONFIG_JSON=$(jq -n --arg token "$NGC_API_KEY" \
  '{
    auths: {
      "nvcr.io": {
        username: "$oauthtoken",
        password: $token,
        auth: ("$oauthtoken:" + $token | @base64)
      }
    }
  }' | base64 -w 0)

# Step 4: Render the template and apply it using kubectl
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: ngc-secret
  namespace: nim-service
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: ${DOCKER_CONFIG_JSON}
---
apiVersion: v1
kind: Secret
metadata:
  name: ngc-api-secret
  namespace: nim-service
type: Opaque
stringData:
  NGC_API_KEY: ${NGC_API_KEY}
EOF

echo "[INFO] Secrets created successfully in namespace 'nim-service'."
