#!/bin/bash
# Destroy what ./install.sh deployed. install.sh copies the base Terraform
# into terraform/_LOCAL and applies it there, so the state is there too.
if [ ! -f ./terraform/_LOCAL/cleanup.sh ]; then
  echo "FAILED: terraform/_LOCAL/cleanup.sh not found. Run this from infra/nvidia-nim after ./install.sh."
  exit 1
fi
cd terraform/_LOCAL
source ./cleanup.sh
