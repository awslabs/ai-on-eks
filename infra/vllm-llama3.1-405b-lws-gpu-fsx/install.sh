#!/bin/bash
# Infra wrapper for the multi-node Llama-3.1-405B FSx-over-EFA blueprint.
# Copies the shared base into a local working dir and applies it with ./terraform/blueprint.tfvars.
# The _LOCAL copy is gitignored (generated, not committed).
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

cd terraform/_LOCAL
source ./install.sh
