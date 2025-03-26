#!/bin/bash
# Copy the base into the folder
cp -r ../base/terraform/* ./terraform

cd terraform
source ./install.sh
