#!/bin/bash
#
# deploy.sh — Terraform lifecycle wrapper for the AI on EKS infrastrucutre stack
#
# Manages S3 remote state backend creation, Terraform init/plan/apply/destroy, and local file cleanup.
#
# Prerequisites:
#   - AWS CLI configured with valid credentials (aws sts get-caller-identity must work)
#   - Terraform installed and available on PATH
#   - jq installed (used to parse AWS CLI JSON output)
#
# Commands:
#   ./deploy.sh bootstrap        Create the S3 bucket for Terraform remote state.
#                                 The bucket is created with versioning, AES-256 encryption,
#                                 and all public access blocked. Safe to run multiple times;
#                                 skips creation if the bucket already exists.
#                                 (default: ai-on-eks-tfstate-<account-id>-<region>)
#
#   ./deploy.sh init [upgrade]   Initialize Terraform with the S3 backend.
#                                 Pass "true" as a second argument to run with -upgrade
#                                 (useful after changing provider versions).
#
#   ./deploy.sh plan             Run terraform plan. Automatically includes
#                                 ../blueprint.tfvars if the file exists.
#
#   ./deploy.sh apply [auto]     Apply the infrastructure. Modules are applied in
#                                 dependency order (VPC, VPC Endpoints, EKS, Karpenter,
#                                 ArgoCD) before a final full apply to catch everything
#                                 else. Pass "false" to disable -auto-approve and get
#                                 an interactive confirmation prompt for each step.
#
#   ./deploy.sh destroy [auto]   Tear down all resources. kubectl_manifest resources
#                                 are destroyed first (except the ALB controller) to
#                                 avoid ordering issues, then the remaining infra is
#                                 destroyed. Pass "false" to disable -auto-approve.
#
#   ./deploy.sh cleanup          Remove local .terraform directories and
#                                 .terraform.lock.hcl files from the working tree.
#
#   --dry-run                     Can be added to any command to print the terraform
#                                 commands instead of executing them.
#                                 e.g. ./deploy.sh plan --dry-run
#
#   --var-file=<path>             Supply an additional .tfvars file. Can be specified
#                                 multiple times. Passed as -var-file to terraform.
#                                 e.g. ./deploy.sh plan --var-file=custom.tfvars
#
# Configuration:
#   The script reads configuration from environment variables and falls back to
#   sensible defaults. If ../blueprint.tfvars exists it is automatically passed
#   as -var-file to plan, apply, and destroy.
#
#   DEPLOYMENT_NAME  — Name of the stack. Resolved in order:
#                      1. terraform output "deployment_name" (if state exists)
#                      2. "name" field in ../blueprint.tfvars or any passed tfvars file via --var-file.
#                      3. DEPLOYMENT_NAME environment variable
#                      4. Falls back to "ai-stack" (default)
#   AWS_REGION       — Target AWS region. Resolved in order:
#                      1. "region" field in ../blueprint.tfvars or any passed tfvars file via --var-file.
#                      2. AWS_REGION envrionment variable
#                      3. Falls back to "us-west-2" (default)
#   TF_STATE_BUCKET  — S3 bucket for remote state
#                      (default: ai-on-eks-tfstate-<account-id>-<region>)
#   TF_STATE_PATH    — Key path inside the bucket
#                      (default: ai-on-eks/<deployment-name>.tfstate)
#
# Typical first-time workflow:
#   ./deploy.sh bootstrap   # one-time: create the state bucket
#   ./deploy.sh init        # initialize terraform
#   ./deploy.sh plan        # preview changes
#   ./deploy.sh apply       # deploy the stack
#
# Tear down:
#   ./deploy.sh destroy     # destroy all resources
#   ./deploy.sh cleanup     # optional: remove local terraform files
#

set -e
# set -x

source ./common.sh

# Parse command line arguments
case "${1:-}" in
  "init")
    echo "Initializing Terraform..."
    TF_INIT_UPGRADE="${2:-false}"
    terraform_init
    ;;
  "plan")
    echo "Planning Terraform changes..."
    terraform_plan
    ;;
  "apply")
    echo "Applying Terraform changes..."
    auto_approve="${2:-true}"
    terraform_apply "$auto_approve"
    ;;
  "destroy")
    echo "Destroying Terraform resources..."
    auto_approve="${2:-true}"
    terraform_destroy "$auto_approve"
    ;;
  "cleanup")
    echo "Cleaning up Terraform local files..."
    terraform_cleanup
    ;;
  "bootstrap")
    echo "Bootstrapping..."
    bootstrap
    ;;
  *)
    echo "Usage: $0 [--dry-run] [--var-file=<path>] {init|plan|apply|destroy|cleanup|bootstrap}"
    echo "  --dry-run              - Print terraform commands instead of executing them"
    echo "  --var-file=<path>      - Additional .tfvars file (can be repeated)"
    echo "  init [upgrade]  - Initialize Terraform with S3 backend (pass 'true' for upgrade)"
    echo "  plan            - Plan Terraform changes"
    echo "  apply           - Apply Terraform changes"
    echo "  destroy         - Destroy Terraform resources"
    echo "  cleanup         - Clean up Terraform local files"
    echo "  bootstrap       - Create S3 bucket for Terraform state"
    exit 1
    ;;
esac
