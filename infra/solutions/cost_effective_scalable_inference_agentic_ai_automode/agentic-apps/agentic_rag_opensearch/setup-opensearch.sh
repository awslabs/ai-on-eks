#!/bin/bash

# Script to create an OpenSearch cluster for agentic RAG application

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
INSTANCE_TYPE="t3.small.search"
INSTANCE_COUNT=1
VOLUME_SIZE=20
VOLUME_TYPE="gp3"
OPENSEARCH_VERSION="2.11"

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
  
  # Get current AWS account ID and user ARN
  ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
  USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
  
  log "AWS Account ID: $ACCOUNT_ID"
  log "User ARN: $USER_ARN"
  
  success "All prerequisites satisfied."
}

# Get default VPC and subnet information
get_vpc_info() {
  log "Getting VPC and subnet information..."
  
  # Get default VPC
  DEFAULT_VPC=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text --region $REGION)
  
  if [ "$DEFAULT_VPC" = "None" ] || [ -z "$DEFAULT_VPC" ]; then
    error "No default VPC found. Please create a VPC first."
  fi
  
  log "Using VPC: $DEFAULT_VPC"
  
  # Get subnets in the default VPC
  SUBNETS=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$DEFAULT_VPC" --query "Subnets[*].SubnetId" --output text --region $REGION)
  
  if [ -z "$SUBNETS" ]; then
    error "No subnets found in VPC $DEFAULT_VPC"
  fi
  
  # Convert to array and take first two subnets
  SUBNET_ARRAY=($SUBNETS)
  SUBNET_IDS="${SUBNET_ARRAY[0]}"
  if [ ${#SUBNET_ARRAY[@]} -gt 1 ]; then
    SUBNET_IDS="$SUBNET_IDS,${SUBNET_ARRAY[1]}"
  fi
  
  log "Using subnets: $SUBNET_IDS"
  
  success "VPC information retrieved successfully."
}
# Create security group for OpenSearch
create_security_group() {
  log "Creating security group for OpenSearch..."
  
  # Check if security group already exists
  EXISTING_SG=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=opensearch-sg" "Name=vpc-id,Values=$DEFAULT_VPC" --query "SecurityGroups[0].GroupId" --output text --region $REGION 2>/dev/null || echo "None")
  
  if [ "$EXISTING_SG" != "None" ] && [ -n "$EXISTING_SG" ]; then
    log "Security group already exists: $EXISTING_SG"
    SECURITY_GROUP_ID=$EXISTING_SG
  else
    # Create security group
    SECURITY_GROUP_ID=$(aws ec2 create-security-group \
      --group-name opensearch-sg \
      --description "Security group for OpenSearch cluster" \
      --vpc-id $DEFAULT_VPC \
      --query "GroupId" \
      --output text \
      --region $REGION)
    
    log "Created security group: $SECURITY_GROUP_ID"
    
    # Add inbound rule for HTTPS (port 443)
    aws ec2 authorize-security-group-ingress \
      --group-id $SECURITY_GROUP_ID \
      --protocol tcp \
      --port 443 \
      --cidr 0.0.0.0/0 \
      --region $REGION
    
    log "Added HTTPS inbound rule to security group"
  fi
  
  success "Security group configured: $SECURITY_GROUP_ID"
}

# Check if OpenSearch domain already exists
check_existing_domain() {
  log "Checking if OpenSearch domain already exists..."
  
  EXISTING_DOMAIN=$(aws opensearch describe-domain --domain-name $DOMAIN_NAME --region $REGION 2>/dev/null || echo "")
  
  if [ -n "$EXISTING_DOMAIN" ]; then
    DOMAIN_STATUS=$(echo "$EXISTING_DOMAIN" | jq -r '.DomainStatus.Processing')
    DOMAIN_ENDPOINT=$(echo "$EXISTING_DOMAIN" | jq -r '.DomainStatus.Endpoint')
    
    if [ "$DOMAIN_STATUS" = "false" ] && [ "$DOMAIN_ENDPOINT" != "null" ]; then
      warn "OpenSearch domain '$DOMAIN_NAME' already exists and is active."
      log "Domain endpoint: https://$DOMAIN_ENDPOINT"
      
      # Update .env file with existing endpoint
      update_env_file "https://$DOMAIN_ENDPOINT"
      
      success "Using existing OpenSearch domain."
      exit 0
    else
      warn "OpenSearch domain '$DOMAIN_NAME' exists but is still processing. Please wait for it to complete."
      exit 1
    fi
  fi
  
  log "No existing domain found. Proceeding with creation..."
}

# Create OpenSearch domain
create_opensearch_domain() {
  log "Creating OpenSearch domain: $DOMAIN_NAME"
  
  # Create the domain configuration
  cat > opensearch-domain-config.json << EOF
{
  "DomainName": "$DOMAIN_NAME",
  "EngineVersion": "OpenSearch_$OPENSEARCH_VERSION",
  "ClusterConfig": {
    "InstanceType": "$INSTANCE_TYPE",
    "InstanceCount": $INSTANCE_COUNT,
    "DedicatedMasterEnabled": false
  },
  "EBSOptions": {
    "EBSEnabled": true,
    "VolumeType": "$VOLUME_TYPE",
    "VolumeSize": $VOLUME_SIZE
  },
  "VPCOptions": {
    "SubnetIds": ["$(echo $SUBNET_IDS | tr ',' '", "')"],
    "SecurityGroupIds": ["$SECURITY_GROUP_ID"]
  },
  "AccessPolicies": "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"$USER_ARN\"},\"Action\":\"es:*\",\"Resource\":\"arn:aws:es:$REGION:$ACCOUNT_ID:domain/$DOMAIN_NAME/*\"}]}",
  "EncryptionAtRestOptions": {
    "Enabled": true
  },
  "NodeToNodeEncryptionOptions": {
    "Enabled": true
  },
  "DomainEndpointOptions": {
    "EnforceHTTPS": true,
    "TLSSecurityPolicy": "Policy-Min-TLS-1-2-2019-07"
  },
  "AdvancedSecurityOptions": {
    "Enabled": false
  }
}
EOF

  # Create the domain
  aws opensearch create-domain \
    --cli-input-json file://opensearch-domain-config.json \
    --region $REGION
  
  log "OpenSearch domain creation initiated..."
  
  # Clean up temporary file
  rm -f opensearch-domain-config.json
  
  success "OpenSearch domain creation request submitted."
}
# Wait for domain to be ready
wait_for_domain() {
  log "Waiting for OpenSearch domain to be ready..."
  
  local max_attempts=60  # 30 minutes (30 seconds * 60)
  local attempt=1
  
  while [ $attempt -le $max_attempts ]; do
    log "Attempt $attempt/$max_attempts: Checking domain status..."
    
    DOMAIN_INFO=$(aws opensearch describe-domain --domain-name $DOMAIN_NAME --region $REGION 2>/dev/null || echo "")
    
    if [ -n "$DOMAIN_INFO" ]; then
      PROCESSING=$(echo "$DOMAIN_INFO" | jq -r '.DomainStatus.Processing')
      ENDPOINT=$(echo "$DOMAIN_INFO" | jq -r '.DomainStatus.Endpoint')
      
      if [ "$PROCESSING" = "false" ] && [ "$ENDPOINT" != "null" ]; then
        success "OpenSearch domain is ready!"
        log "Domain endpoint: https://$ENDPOINT"
        DOMAIN_ENDPOINT="https://$ENDPOINT"
        return 0
      fi
    fi
    
    log "Domain is still processing. Waiting 30 seconds..."
    sleep 30
    ((attempt++))
  done
  
  error "Domain failed to become ready within the expected time. Please check the AWS console."
}

# Update .env file with the new endpoint
update_env_file() {
  local endpoint=$1
  log "Updating .env file with OpenSearch endpoint..."
  
  if [ -f ".env" ]; then
    # Update existing OPENSEARCH_ENDPOINT line or add it
    if grep -q "OPENSEARCH_ENDPOINT=" .env; then
      sed -i.bak "s|OPENSEARCH_ENDPOINT=.*|OPENSEARCH_ENDPOINT=$endpoint|" .env
    else
      echo "OPENSEARCH_ENDPOINT=$endpoint" >> .env
    fi
    
    # Update AWS_REGION if not present
    if ! grep -q "AWS_REGION=" .env; then
      echo "AWS_REGION=$REGION" >> .env
    fi
    
    success ".env file updated with OpenSearch endpoint."
  else
    # Create new .env file
    cat > .env << EOF
OPENAI_API_KEY=your-openai-api-key
OPENAI_BASE_URL=your-model-endpoint
OPENSEARCH_ENDPOINT=$endpoint
AWS_REGION=$REGION
EOF
    success "Created .env file with OpenSearch endpoint."
  fi
}

# Verify the setup
verify_setup() {
  log "Verifying OpenSearch setup..."
  
  # Test connection (basic check)
  log "Testing OpenSearch connectivity..."
  
  # Get domain info
  DOMAIN_INFO=$(aws opensearch describe-domain --domain-name $DOMAIN_NAME --region $REGION)
  
  if [ $? -eq 0 ]; then
    success "OpenSearch domain is accessible via AWS CLI."
    
    # Display domain information
    echo "$DOMAIN_INFO" | jq '.DomainStatus | {
      DomainName: .DomainName,
      Endpoint: .Endpoint,
      Processing: .Processing,
      InstanceType: .ClusterConfig.InstanceType,
      InstanceCount: .ClusterConfig.InstanceCount,
      VolumeSize: .EBSOptions.VolumeSize
    }'
  else
    error "Failed to verify OpenSearch domain."
  fi
}

# Main execution
main() {
  log "Starting OpenSearch cluster setup for agentic RAG application..."
  
  check_prerequisites
  get_vpc_info
  create_security_group
  check_existing_domain
  create_opensearch_domain
  wait_for_domain
  update_env_file "$DOMAIN_ENDPOINT"
  verify_setup
  
  success "OpenSearch cluster setup completed successfully!"
  log "Domain Name: $DOMAIN_NAME"
  log "Endpoint: $DOMAIN_ENDPOINT"
  log "Region: $REGION"
  log ""
  log "Next steps:"
  log "1. Update your .env file with the correct OPENAI_API_KEY and OPENAI_BASE_URL"
  log "2. Run 'pnpm install' to install dependencies"
  log "3. Run 'pnpm embed-knowledge' to embed your knowledge documents"
  log "4. Run 'pnpm dev' to start the agentic RAG application"
}

# Execute main function
main
