#!/bin/bash
# Copy the base into the folder
mkdir -p ./terraform/_LOCAL
cp -r ../../base/terraform/* ./terraform/_LOCAL

cd terraform/_LOCAL
source ./install.sh
