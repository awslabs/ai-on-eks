#!/bin/bash
# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

cd terraform/_LOCAL
source ./install.sh

echo "Run the following command to create the HF token secret:"
echo ""
echo "  kubectl create secret generic hf-token-secret \\"
echo "    --from-literal=HF_TOKEN=<HF TOKEN> \\"
echo "    -n dynamo-system"
