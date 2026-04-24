#!/usr/bin/env bash

# https://github.com/aws/eks-charts/blob/master/hack/update-efa-instance-types.sh

set -euo pipefail

# This script requires yq >= v4.52, install it here:
# https://github.com/mikefarah/yq/?tab=readme-ov-file#install

# Hard code just preview instance types or ones that may not show up in Describe responses
ALL_TYPES=("p6e-gb300.36xlarge" "p6e-gb200.36xlarge" "trn2-ac.24xlarge" "trn2u-ac.24xlarge" "trn2u.48xlarge")

PROJECT_ROOT=$(git rev-parse --show-toplevel)
DRANET_HELM_VALUES_DIRECTORY="${PROJECT_ROOT}/infra/base/terraform/helm-values"

# Get list of opted-in regions
REGIONS=()
while IFS= read -r line; do
  [[ -n "$line" ]] && REGIONS+=("$line")
done < <(
  aws ec2 describe-regions \
    --query 'Regions[?OptInStatus==`opt-in-not-required` || OptInStatus==`opted-in`].RegionName' \
    --output text \
  | tr '\t' '\n'
)

# Fetch instance types from each region
for REGION in "${REGIONS[@]}"; do
  echo "Getting EFA instance types in $REGION"
  TYPES=()
  while IFS= read -r line; do
    [[ -n "$line" ]] && TYPES+=("$line")
  done < <(
    aws ec2 describe-instance-types \
      --region "$REGION" \
      --filters "Name=network-info.efa-supported,Values=true" \
      --query 'InstanceTypes[*].InstanceType' \
      --output text \
    | tr '\t' '\n' \
    | sed '/^$/d')

  ALL_TYPES+=("${TYPES[@]}")
done

# Build a yq array, then deduplicate + sort with yq builtins
export YQ_ARRAY=$(printf -- '- %s\n' "${ALL_TYPES[@]}" | yq 'unique | sort')

# Update EFA instance types for DRANET driver
VALUES_FILE="${DRANET_HELM_VALUES_DIRECTORY}/dranet-driver.yaml"
yq eval -i '(.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[] | select(.key == "node.kubernetes.io/instance-type")).values = env(YQ_ARRAY)' "$VALUES_FILE"
