#!/bin/bash
# Install KRO on the cluster so the AgentSandbox ResourceGraphDefinition
# below can register itself as a real CRD. KRO is at kubernetes-sigs/kro
# and reached v0.9.x as of April 2026 — still pre-1.0 but stabilizing.
# When kro graduates to 1.0 / GA, bump KRO_VERSION and re-validate
# rgd-agent-sandbox.yaml against any breaking schema changes.
set -euo pipefail

KRO_VERSION="${KRO_VERSION:-0.9.1}"
kubectl apply -f "https://github.com/kubernetes-sigs/kro/releases/download/v${KRO_VERSION}/kro.yaml"
kubectl -n kro wait --for=condition=Available deployment --all --timeout=3m
