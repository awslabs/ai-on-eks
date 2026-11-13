#!/bin/bash

set -e

# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../base/terraform/* ./terraform/_LOCAL

cd terraform/_LOCAL
source ./common.sh

terraform_cleanup
terraform_init
terraform_destroy
