#!/bin/bash

# --------------------------------------------------------------------
# Script: install.sh
# Purpose: Bootstraps local Terraform setup by:
#   1. Copying base infrastructure templates
#   2. Executing the local Terraform install script
# --------------------------------------------------------------------

set -e  # Exit immediately if a command exits with a non-zero status

echo "[INFO] Starting Terraform bootstrap process..."

# --------------------------------------------------------------------
# Step 1: Ensure terraform/ directory exists
# --------------------------------------------------------------------
mkdir -p ./terraform

# --------------------------------------------------------------------
# Step 2: Execute install.sh script inside the terraform directory
# --------------------------------------------------------------------
cd terraform

if [ -f "./install.sh" ]; then
  echo "[INFO] Running terraform/install.sh..."
  source ./install.sh
else
  echo "[ERROR] install.sh not found in ./terraform. Exiting."
  exit 1
fi
