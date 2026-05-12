#!/bin/bash
# Install KRO on the cluster so the AgentSandbox ResourceGraphDefinition
# below can register itself as a real CRD. KRO is at kubernetes-sigs/kro
# and reached v0.9.x as of April 2026 — still pre-1.0 but stabilizing.
# When kro graduates to 1.0 / GA, bump KRO_VERSION and re-validate
# rgd-agent-sandbox.yaml against any breaking schema changes.
#
# KRO v0.9+ ships only as a Helm chart from an OCI registry
# (registry.k8s.io/kro/charts/kro). Older raw-manifest release
# bundles are no longer published on the GitHub releases page.
set -euo pipefail

KRO_VERSION="${KRO_VERSION:-0.9.1}"
helm upgrade --install kro oci://registry.k8s.io/kro/charts/kro \
    --version "${KRO_VERSION}" \
    --namespace kro-system \
    --create-namespace \
    --wait --timeout 3m
