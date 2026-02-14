#!/bin/bash

# Script to cleanup OpenSearch cluster and related resources

set -e

# Color codes for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to display messages with timestamp
log() {
  echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
  echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
  echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
  exit 1
}

success() {
  echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Configuration variables
DOMAIN_NAME="agentic-rag-opensearch"
REGION="us-east-1"
SECURITY_GROUP_NAME="opensearch-sg"

# Check prerequisites
check_prerequisites() {
  log "Checking prerequisites..."
  
  # Check AWS CLI
  if ! command -v aws &> /dev/null; then
    error "AWS CLI is not installed. Please install it first."
  fi
  
  # Check AWS credentials
  if ! aws sts get-caller-identity &> /dev/null; then
    error "AWS credentials not configured or invalid. Please configure AWS CLI."
  fi
  
  success "All prerequisites satisfied."
}

# Delete OpenSearch domain
delete_opensearch_domain() {
  log "Checking for OpenSearch domain: $DOMAIN_NAME"
  
  EXISTING_DOMAIN=$(aws opensearch describe-domain --domain-name $DOMAIN_NAME --region $REGION 2>/dev/null || echo "")
  
  if [ -n "$EXISTING_DOMAIN" ]; then
    log "Deleting OpenSearch domain: $DOMAIN_NAME"
    
    aws opensearch delete-domain \
      --domain-name $DOMAIN_NAME \
      --region $REGION
    
    log "OpenSearch domain deletion initiated. This may take several minutes..."
    
    # Wait for deletion to complete
    local max_attempts=30  # 15 minutes
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
      log "Attempt $attempt/$max_attempts: Checking if domain is deleted..."
      
      DOMAIN_CHECK=$(aws opensearch describe-domain --domain-name $DOMAIN_NAME --region $REGION 2>/dev/null || echo "DELETED")
      
      if [ "$DOMAIN_CHECK" = "DELETED" ]; then
        success "OpenSearch domain deleted successfully!"
        break
      fi
      
      log "Domain still exists. Waiting 30 seconds..."
      sleep 30
      ((attempt++))
    done
    
    if [ $attempt -gt $max_attempts ]; then
      warn "Domain deletion is taking longer than expected. Please check AWS console."
    fi
  else
    log "OpenSearch domain '$DOMAIN_NAME' not found. Skipping deletion."
  fi
}

# Delete security group
delete_security_group() {
  log "Checking for security group: $SECURITY_GROUP_NAME"
  
  # Get default VPC
  DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text --region $REGION)
  
  if [ "$DEFAULT_VPC" = "None" ] || [ -z "$DEFAULT_VPC" ]; then
    log "No default VPC found. Skipping security group deletion."
    return
  fi
  
  EXISTING_SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP_NAME" "Name=vpc-id,Values=$DEFAULT_VPC" --query "SecurityGroups[0].GroupId" --output text --region $REGION 2>/dev/null || echo "None")
  
  if [ "$EXISTING_SG" != "None" ] && [ -n "$EXISTING_SG" ]; then
    log "Deleting security group: $EXISTING_SG"
    
    # Try to delete the security group
    if aws ec2 delete-security-group --group-id $EXISTING_SG --region $REGION 2>/dev/null; then
      success "Security group deleted successfully!"
    else
      warn "Failed to delete security group. It may still be in use or have dependencies."
      log "You may need to delete it manually from the AWS console."
    fi
  else
    log "Security group '$SECURITY_GROUP_NAME' not found. Skipping deletion."
  fi
}

# Clean up .env file
cleanup_env_file() {
  log "Cleaning up .env file..."
  
  if [ -f ".env" ]; then
    # Remove or comment out OPENSEARCH_ENDPOINT
    if grep -q "OPENSEARCH_ENDPOINT=" .env; then
      sed -i.bak 's/^OPENSEARCH_ENDPOINT=/#OPENSEARCH_ENDPOINT=/' .env
      success ".env file updated (OPENSEARCH_ENDPOINT commented out)."
    else
      log "OPENSEARCH_ENDPOINT not found in .env file."
    fi
  else
    log ".env file not found. Nothing to clean up."
  fi
}

# Main execution
main() {
  log "Starting OpenSearch cluster cleanup..."
  
  warn "This will delete the OpenSearch domain and related resources."
  read -p "Are you sure you want to continue? (y/N): " -n 1 -r
  echo
  
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Cleanup cancelled."
    exit 0
  fi
  
  check_prerequisites
  delete_opensearch_domain
  delete_security_group
  cleanup_env_file
  
  success "OpenSearch cluster cleanup completed!"
  log "All resources have been cleaned up."
  log "Note: It may take a few minutes for all AWS resources to be fully removed."
}

# Execute main function
main
