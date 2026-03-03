#!/bin/bash

# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../../base/terraform/* ./terraform/_LOCAL
cp ./terraform/s3-workshop.tf ./terraform/_LOCAL/
cp ./terraform/pull.tf ./terraform/_LOCAL/
cp ./terraform/grafana.tf ./terraform/_LOCAL/
cp -r ./terraform/grafana-dashboards ./terraform/_LOCAL/
cp ./terraform/blueprint.tfvars ./terraform/_LOCAL/

cd terraform/_LOCAL
source ./install.sh
