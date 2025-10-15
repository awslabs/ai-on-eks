#!/bin/bash

# NVIDIA Dynamo cleanup script - removes all deployments and infrastructure
set -euo pipefail

# Ensure we're in the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
CLUSTER_NAME="dynamo-on-eks"
REGION="us-west-2"
NAMESPACE="dynamo-cloud"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Confirmation prompt (skip if --force flag is provided)
if [[ "${1:-}" != "--force" ]]; then
    echo ""
    warn "⚠️  This will completely remove all NVIDIA Dynamo infrastructure and deployments!"
    echo "   Cluster: ${CLUSTER_NAME}"
    echo "   Region: ${REGION}"
    echo "   This action cannot be undone."
    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        info "Cleanup cancelled by user"
        exit 0
    fi
fi

info "Starting NVIDIA Dynamo cleanup for ${CLUSTER_NAME}..."

# Check if cluster exists and is accessible
info "Checking cluster accessibility..."
CLUSTER_ACCESSIBLE=false

# Try to update kubeconfig
if aws eks update-kubeconfig --region ${REGION} --name ${CLUSTER_NAME} 2>/dev/null; then
    info "Successfully updated kubeconfig"
    # Test if we can actually connect to the cluster
    if kubectl cluster-info --request-timeout=10s >/dev/null 2>&1; then
        info "Cluster is accessible"
        CLUSTER_ACCESSIBLE=true
    else
        warn "Cluster exists but is not accessible (may be in deletion process)"
        CLUSTER_ACCESSIBLE=false
    fi
else
    warn "Failed to update kubeconfig - cluster may not exist or be inaccessible"
    CLUSTER_ACCESSIBLE=false
fi

# Phase 1: Clean up Dynamo graphs manually (only if cluster is accessible)
if [ "$CLUSTER_ACCESSIBLE" = true ]; then
    info "Phase 1: Cleaning up Dynamo custom resources..."
else
    info "Phase 1: Skipping Kubernetes resource cleanup (cluster not accessible)"
fi

if [ "$CLUSTER_ACCESSIBLE" = true ]; then
    # Delete DynamoGraphDeployments first
    info "Deleting DynamoGraphDeployments..."
    if kubectl get dynamographdeployment -A --no-headers -o custom-columns=":metadata.name,:metadata.namespace" 2>/dev/null | grep -q .; then
        kubectl get dynamographdeployment -A --no-headers -o custom-columns=":metadata.name,:metadata.namespace" 2>/dev/null | while read name namespace; do
            if [ -n "$name" ] && [ -n "$namespace" ]; then
                info "Removing finalizers and deleting DynamoGraphDeployment: $name in namespace $namespace"
                kubectl patch dynamographdeployment $name -n $namespace --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || warn "Failed to patch finalizers"
                kubectl delete dynamographdeployment $name -n $namespace --ignore-not-found=true || warn "Failed to delete DynamoGraphDeployment: $name"
            fi
        done

        # Wait for DynamoGraphDeployments to be fully deleted
        info "Waiting for DynamoGraphDeployments to be fully deleted..."
        while kubectl get dynamographdeployment -A --no-headers 2>/dev/null | grep -q .; do
            info "Still waiting for DynamoGraphDeployments to complete deletion..."
            sleep 5
        done
        info "All DynamoGraphDeployments deleted"
    else
        info "No DynamoGraphDeployments found"
    fi

    # Delete DynamoComponentDeployments
    info "Deleting DynamoComponentDeployments..."
    if kubectl get dynamocomponentdeployment -A --no-headers -o custom-columns=":metadata.name,:metadata.namespace" 2>/dev/null | grep -q .; then
        kubectl get dynamocomponentdeployment -A --no-headers -o custom-columns=":metadata.name,:metadata.namespace" 2>/dev/null | while read name namespace; do
            if [ -n "$name" ] && [ -n "$namespace" ]; then
                info "Removing finalizers and deleting DynamoComponentDeployment: $name in namespace $namespace"
                kubectl patch dynamocomponentdeployment $name -n $namespace --type='merge' -p='{"metadata":{"finalizers":null}}' 2>/dev/null || warn "Failed to patch finalizers"
                kubectl delete dynamocomponentdeployment $name -n $namespace --ignore-not-found=true || warn "Failed to delete DynamoComponentDeployment: $name"
            fi
        done

        # Wait for DynamoComponentDeployments to be fully deleted
        info "Waiting for DynamoComponentDeployments to be fully deleted..."
        while kubectl get dynamocomponentdeployment -A --no-headers 2>/dev/null | grep -q .; do
            info "Still waiting for DynamoComponentDeployments to complete deletion..."
            sleep 5
        done
        info "All DynamoComponentDeployments deleted"
    else
        info "No DynamoComponentDeployments found"
    fi
else
    info "Skipping Dynamo custom resource cleanup (cluster not accessible)"
fi

# Phase 2: Clean up other conflicting resources
info "Phase 2: Cleaning up other conflicting resources..."

# Remove the existing CloudWatch log group
info "Removing existing CloudWatch log group..."
aws logs delete-log-group --log-group-name "/aws/eks/${CLUSTER_NAME}/cluster" --region ${REGION} 2>/dev/null || info "Log group not found or already deleted"


# Phase 3: Run base cleanup and remove _LOCAL directory
info "Phase 3: Running Terraform cleanup..."

TERRAFORM_SUCCESS=false

if [ -d "terraform/_LOCAL" ]; then
    info "Found terraform/_LOCAL directory, running cleanup..."

    # Save current directory
    ORIGINAL_DIR=$(pwd)
    cd terraform/_LOCAL

    if [ -f "./cleanup.sh" ]; then
        info "Running base cleanup script..."
        if ./cleanup.sh; then
            info "Base cleanup script completed successfully"
            TERRAFORM_SUCCESS=true
        else
            error "Base cleanup script failed!"
            TERRAFORM_SUCCESS=false
        fi
    else
        warn "Base cleanup script not found, attempting manual cleanup..."
        if [ -f "terraform.tfstate" ]; then
            info "Running terraform destroy..."
            if terraform destroy -auto-approve -var-file=../blueprint.tfvars; then
                info "Terraform destroy completed successfully"
                TERRAFORM_SUCCESS=true
            else
                error "Terraform destroy failed!"
                TERRAFORM_SUCCESS=false
            fi
        else
            warn "No terraform.tfstate found, assuming infrastructure was already destroyed"
            TERRAFORM_SUCCESS=true
        fi
    fi

    # Return to original directory
    cd "$ORIGINAL_DIR"

    # Only remove the directory if terraform cleanup was successful
    if [ "$TERRAFORM_SUCCESS" = true ]; then
        info "Removing terraform/_LOCAL directory..."
        rm -rf terraform/_LOCAL
        info "Terraform working directory cleaned up"
    else
        error "Terraform cleanup failed - preserving terraform/_LOCAL directory for troubleshooting"
        error "Please check the terraform state and resolve any issues, then re-run cleanup.sh"
    fi
else
    warn "terraform/_LOCAL directory not found, skipping Terraform cleanup"
    TERRAFORM_SUCCESS=true  # No terraform to clean up
fi

# Final status report
echo ""
if [ "$TERRAFORM_SUCCESS" = true ]; then
    info "✅ NVIDIA Dynamo cleanup completed successfully!"
    echo ""
    echo "Cleaned up:"
    if [ "$CLUSTER_ACCESSIBLE" = true ]; then
        echo "  ✓ DynamoGraphDeployments and DynamoComponentDeployments"
    else
        echo "  ⚠ Kubernetes resources (skipped - cluster not accessible)"
    fi
    echo "  ✓ Conflicting CloudWatch log groups"
    echo "  ✓ Terraform infrastructure and working directory"
    echo ""
    if [ "$CLUSTER_ACCESSIBLE" = true ]; then
        echo "Note: The dynamo-cloud namespace and ArgoCD applications will be"
        echo "      automatically cleaned up by Terraform when it destroys the"
        echo "      ArgoCD applications (CreateNamespace=true handles this)."
    else
        echo "Note: Kubernetes resources were skipped because the cluster was not accessible."
        echo "      This is normal if the infrastructure was already destroyed."
    fi
else
    error "❌ NVIDIA Dynamo cleanup completed with errors!"
    echo ""
    echo "Cleaned up:"
    if [ "$CLUSTER_ACCESSIBLE" = true ]; then
        echo "  ✓ DynamoGraphDeployments and DynamoComponentDeployments"
    else
        echo "  ⚠ Kubernetes resources (skipped - cluster not accessible)"
    fi
    echo "  ✓ Conflicting CloudWatch log groups"
    echo "  ❌ Terraform infrastructure cleanup failed"
    echo ""
    echo "⚠️  IMPORTANT: Terraform destroy failed!"
    echo "   The terraform/_LOCAL directory has been preserved for troubleshooting."
    echo "   Please:"
    echo "   1. Check the terraform state in terraform/_LOCAL/"
    echo "   2. Resolve any resource conflicts or dependencies"
    echo "   3. Re-run this cleanup script: ./cleanup.sh"
    echo ""
    echo "   Common issues:"
    echo "   - Resources still in use by other services"
    echo "   - Network dependencies (VPC, subnets, security groups)"
    echo "   - IAM roles or policies still attached"
    echo "   - Load balancers or other AWS resources not properly cleaned up"
    echo ""
    exit 1
fi
