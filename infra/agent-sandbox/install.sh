#!/bin/bash
# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

# Copy custom agent-sandbox Karpenter nodepools (e.g. the SOCI lazy-load
# pool) into the karpenter-resources dir so the base module renders + applies
# them alongside the defaults. Same pattern as infra/static-nodepool.
if compgen -G "./nodepools/*.yaml" > /dev/null; then
  cp nodepools/*.yaml ./terraform/_LOCAL/karpenter-resources/karpenter/
fi

cd terraform/_LOCAL
source ./install.sh
