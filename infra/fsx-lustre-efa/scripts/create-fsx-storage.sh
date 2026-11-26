#!/bin/bash
# Creates the FSx for Lustre (EFA) StorageClass and PVC from the blueprint's Terraform outputs.
#
# The script prints the manifests first and asks before applying them. FSx for Lustre bills while the
# file system exists, so read the manifests. Deleting the PVC deletes the file system when the
# reclaim policy is Delete.
#
# Usage: create-fsx-storage.sh -z <availability-zone> [-c GiB] [-t MBps/TiB] [-n namespace]
#                              [-s name] [-r Delete|Retain] [-d terraform-dir] [-y]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AZ=""
CAPACITY_GIB="4800"
THROUGHPUT="1000"
NAMESPACE="default"
NAME=""
RECLAIM_POLICY="Delete"
TF_DIR="$SCRIPT_DIR/../terraform/_LOCAL"
ASSUME_YES="false"

usage() { sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

while getopts "z:c:t:n:s:r:d:yh" opt; do
  case "$opt" in
    z) AZ="$OPTARG" ;;
    c) CAPACITY_GIB="$OPTARG" ;;
    t) THROUGHPUT="$OPTARG" ;;
    n) NAMESPACE="$OPTARG" ;;
    s) NAME="$OPTARG" ;;
    r) RECLAIM_POLICY="$OPTARG" ;;
    d) TF_DIR="$OPTARG" ;;
    y) ASSUME_YES="true" ;;
    h) usage 0 ;;
    *) usage ;;
  esac
done

for tool in terraform jq envsubst kubectl; do
  command -v "$tool" >/dev/null || { echo "Missing required tool: $tool" >&2; exit 1; }
done
[ -n "$AZ" ] || { echo "-z <availability-zone> is required" >&2; usage; }
case "$RECLAIM_POLICY" in Delete | Retain) ;; *) echo "-r must be Delete or Retain" >&2; exit 1 ;; esac
[[ "$CAPACITY_GIB" =~ ^[0-9]+$ ]] || { echo "-c must be a number of GiB" >&2; exit 1; }
[[ "$THROUGHPUT" =~ ^(125|250|500|1000)$ ]] || { echo "-t must be one of 125, 250, 500, 1000" >&2; exit 1; }

OUTPUTS="$(terraform -chdir="$TF_DIR" output -json)"
CLUSTER_NAME="$(jq -er '.deployment_name.value' <<<"$OUTPUTS")"
SG_ID="$(jq -er '.cluster_primary_security_group_id.value' <<<"$OUTPUTS")"
if ! SUBNET_ID="$(jq -er --arg az "$AZ" '.secondary_subnet_by_az.value[$az]' <<<"$OUTPUTS")"; then
  echo "No secondary subnet for AZ '$AZ'. Available AZs:" >&2
  jq -r '.secondary_subnet_by_az.value | keys[]' <<<"$OUTPUTS" >&2
  exit 1
fi

STORAGE_NAME="${NAME:-fsx-lustre-efa-$AZ}"
export CLUSTER_NAME SG_ID SUBNET_ID STORAGE_NAME NAMESPACE THROUGHPUT CAPACITY_GIB RECLAIM_POLICY

MANIFEST="$(envsubst '${CLUSTER_NAME} ${SG_ID} ${SUBNET_ID} ${STORAGE_NAME} ${NAMESPACE} ${THROUGHPUT} ${CAPACITY_GIB} ${RECLAIM_POLICY}' \
  <"$SCRIPT_DIR/../examples/storageclass-pvc.yaml")"

echo "$MANIFEST"
echo
echo "AZ: $AZ  subnet: $SUBNET_ID  security group: $SG_ID"
if [ "$ASSUME_YES" != "true" ]; then
  read -r -p "Apply these manifests to the current kubectl context ($(kubectl config current-context))? [y/N] " answer
  [ "$answer" = "y" ] || { echo "Not applied."; exit 0; }
fi
kubectl apply -f - <<<"$MANIFEST"
echo "Watch the file system come up with: kubectl get pvc -n $NAMESPACE $STORAGE_NAME -w"
