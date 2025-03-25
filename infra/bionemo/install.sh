#!/bin/bash
# Copy the base infrastructure into the folder
cp -r ../base/terraform/* ./terraform

cd terraform
source ./install.sh
