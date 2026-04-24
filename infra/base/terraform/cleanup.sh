#!/bin/bash

TERRAFORM_COMMAND="terraform destroy -auto-approve"
CLUSTERNAME="ai-stack"
REGION="region"

# Get the deployment_id from terraform output
DEPLOYMENT_NAME=$(terraform output -raw deployment_name)

# Check if blueprint.tfvars exists
if [ -f "../blueprint.tfvars" ]; then
  TERRAFORM_COMMAND="$TERRAFORM_COMMAND -var-file=../blueprint.tfvars"
  CLUSTERNAME="$(echo "var.name" | terraform console -var-file=../blueprint.tfvars | tr -d '"')"
  REGION="$(echo "var.region" | terraform console -var-file=../blueprint.tfvars | tr -d '"')"
fi
echo "Destroying Terraform $CLUSTERNAME"
echo "Destroying RayService..."

# Delete the Ingress/SVC before removing the addons
TMPFILE=$(mktemp)
terraform output -raw configure_kubectl > "$TMPFILE"
# check if TMPFILE contains the string "No outputs found"
if [[ ! $(cat $TMPFILE) == *"No outputs found"* ]]; then
  source "$TMPFILE"
  kubectl delete rayjob -A --all
  kubectl delete rayservice -A --all
else
  echo "No outputs found, skipping kubectl delete"
fi


# List of Terraform modules to destroy in sequence
targets=($(terraform state list | grep "kubectl_manifest\." | grep -v "kubectl_manifest.aws_load_balancer_controller"))

# Destroy all kubectl_manifest resources at once (excluding aws_load_balancer_controller)
if [ ${#targets[@]} -gt 0 ]; then
  echo "Destroying kubectl_manifest resources..."
  target_args=""
  for target in "${targets[@]}"; do
    target_args="$target_args -target=$target"
  done

  destroy_output=$($TERRAFORM_COMMAND $target_args 2>&1 | tee /dev/tty)
  if [[ ${PIPESTATUS[0]} -eq 0 && $destroy_output == *"Destroy complete"* ]]; then
    echo "SUCCESS: Terraform destroy of kubectl_manifest resources completed successfully"
  else
    echo "FAILED: Terraform destroy of kubectl_manifest resources failed"
    exit 1
  fi
fi

## Final destroy to catch any remaining resources
echo "Destroying remaining resources..."
destroy_output=$($TERRAFORM_COMMAND -var="region=$REGION" 2>&1 | tee /dev/tty)
if [[ ${PIPESTATUS[0]} -eq 0 && $destroy_output == *"Destroy complete"* ]]; then
  echo "SUCCESS: Terraform destroy of all modules completed successfully"
else
  echo "FAILED: Terraform destroy of all modules failed"
  exit 1
fi

echo "Cleaning up PVCs and EBS volumes for deployment: $DEPLOYMENT_NAME"

# Get the list of EBS volumes with the Blueprint tag
VOLUME_IDS=$(aws ec2 describe-volumes --region "$REGION" --filters "Name=tag:kubernetes.io/cluster/${DEPLOYMENT_NAME},Values=owned" --query "Volumes[].VolumeId" --output text | tr '\t' '\n')

if [ -n "$VOLUME_IDS" ]; then
  while IFS= read -r volume_id; do
    # Get the PVC name from the volume tags
    PVC_NAME=$(aws ec2 describe-volumes --region "$REGION" --volume-ids "$volume_id" --query "Volumes[0].Tags[?Key=='kubernetes.io/created-for/pvc/name'].Value" --output text)
    PVC_NAMESPACE=$(aws ec2 describe-volumes --region "$REGION" --volume-ids "$volume_id" --query "Volumes[0].Tags[?Key=='kubernetes.io/created-for/pvc/namespace'].Value" --output text)

    echo "Deleting EBS volume: $volume_id, PVC: ${PVC_NAME}, Namespace: ${PVC_NAMESPACE}"
    aws ec2 delete-volume --region "$REGION" --volume-id "$volume_id"
  done <<< "$VOLUME_IDS"
else
  echo "No EBS volumes found for deployment : $DEPLOYMENT_NAME"
fi
