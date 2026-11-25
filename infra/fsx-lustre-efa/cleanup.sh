#!/bin/bash

set -e

echo "This destroys the cluster. Delete every PersistentVolumeClaim created with the"
echo "fsx-lustre-efa StorageClass first (kubectl delete pvc ...), otherwise the FSx"
echo "file systems created by the CSI driver are left behind and keep billing."
read -r -p "Have you deleted the FSx PVCs? [y/N] " answer
[ "$answer" = "y" ] || exit 1

# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL
cp nodepools/*.yaml ./terraform/_LOCAL/karpenter-resources/karpenter/

cd terraform/_LOCAL
source ./common.sh

terraform_cleanup
terraform_init
terraform_destroy
