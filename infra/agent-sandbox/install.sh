#!/bin/bash
# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

# Copy custom Karpenter nodepools (gvisor, kata-fc)
cp nodepools/*.yaml ./terraform/_LOCAL/karpenter-resources/karpenter/

cd terraform/_LOCAL
source ./install.sh
