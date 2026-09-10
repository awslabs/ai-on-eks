#!/bin/bash
# Cleanup script - deletes all resources to stop billing
set -e

REGION=${AWS_REGION:-us-west-2}
CLUSTER_NAME=${CLUSTER_NAME:-cache-routing-benchmark}

echo "=== Cleaning up Cache-Aware Routing resources ==="
echo "This will DELETE the EKS cluster and all associated resources."
read -p "Continue? (y/N): " confirm
if [ "$confirm" != "y" ]; then echo "Aborted."; exit 0; fi

echo "Deleting cluster $CLUSTER_NAME in $REGION..."
eksctl delete cluster --name $CLUSTER_NAME --region $REGION

echo "=== Cleanup complete. All resources deleted. ==="
