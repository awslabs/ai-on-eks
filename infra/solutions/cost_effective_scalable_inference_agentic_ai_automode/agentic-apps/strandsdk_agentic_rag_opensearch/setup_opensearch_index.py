#!/usr/bin/env python3
"""
OpenSearch Index Setup Script
Creates a vector index for embeddings with proper mapping
"""

import os
import json
import logging
import time
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import ConnectionError, ConnectionTimeout
import boto3
from requests_aws4auth import AWS4Auth

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_role_trust_policy(service_account_role_arn):
    """Update the service account role trust policy to allow the current role to assume it"""
    try:
        iam = boto3.client('iam')
        sts = boto3.client('sts')
        
        # Get current identity
        identity = sts.get_caller_identity()
        current_arn = identity['Arn']
        
        # Extract the base role ARN from assumed role ARN
        if ':assumed-role/' in current_arn:
            # Convert from arn:aws:sts::account:assumed-role/role-name/session-name
            # to arn:aws:iam::account:role/role-name
            parts = current_arn.split(':')
            account = parts[4]
            role_info = parts[5].split('/')
            role_name = role_info[1]  # Get role name from assumed-role/role-name/session-name
            current_role_arn = f"arn:aws:iam::{account}:role/{role_name}"
        else:
            current_role_arn = current_arn
            
        logger.info(f"Current role ARN: {current_role_arn}")
        
        # Extract role name from service account ARN
        role_name = service_account_role_arn.split('/')[-1]
        
        # Define the updated trust policy
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "Service": "pods.eks.amazonaws.com"
                    },
                    "Action": [
                        "sts:AssumeRole",
                        "sts:TagSession"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": current_role_arn
                    },
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Update the role trust policy
        response = iam.update_assume_role_policy(
            RoleName=role_name,
            PolicyDocument=json.dumps(trust_policy)
        )
        logger.info(f"Successfully updated role trust policy for {role_name}")
        logger.info(f"Added permission for {current_role_arn} to assume the role")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to update role trust policy: {e}")
        return False

def update_opensearch_access_policy(domain_name, region, service_account_role_arn):
    """Update OpenSearch domain access policy to include the service account role"""
    try:
        opensearch_client = boto3.client('opensearch', region_name=region)
        
        # Get current domain configuration
        response = opensearch_client.describe_domain(DomainName=domain_name)
        domain_config = response['DomainStatus']
        
        # Create updated access policy
        access_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": service_account_role_arn
                    },
                    "Action": "es:*",
                    "Resource": f"arn:aws:es:{region}:{boto3.client('sts').get_caller_identity()['Account']}:domain/{domain_name}/*"
                },
                {
                    "Effect": "Allow",
                    "Principal": {
                        "AWS": "*"
                    },
                    "Action": [
                        "es:ESHttpGet",
                        "es:ESHttpPost",
                        "es:ESHttpPut",
                        "es:ESHttpDelete",
                        "es:ESHttpHead"
                    ],
                    "Resource": f"arn:aws:es:{region}:{boto3.client('sts').get_caller_identity()['Account']}:domain/{domain_name}/*",
                    "Condition": {
                        "IpAddress": {
                            "aws:SourceIp": "0.0.0.0/0"
                        }
                    }
                }
            ]
        }
        
        # Update the domain access policy
        update_response = opensearch_client.update_domain_config(
            DomainName=domain_name,
            AccessPolicies=json.dumps(access_policy)
        )
        
        logger.info(f"Successfully updated OpenSearch domain access policy for {domain_name}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to update OpenSearch domain access policy: {e}")
        return False

def configure_opensearch_iam_access(domain_name, region, service_account_role_arn):
    """Configure fine-grained access control to use IAM instead of internal user database"""
    try:
        opensearch_client = boto3.client('opensearch', region_name=region)
        
        # Update domain configuration to disable internal user database and enable IAM
        update_response = opensearch_client.update_domain_config(
            DomainName=domain_name,
            AdvancedSecurityOptions={
                'Enabled': True,
                'InternalUserDatabaseEnabled': False,
                'MasterUserOptions': {
                    'MasterUserARN': service_account_role_arn
                }
            }
        )
        
        logger.info(f"Successfully configured IAM-based access for {domain_name}")
        logger.info(f"Set master user to: {service_account_role_arn}")
        logger.info("Domain update initiated - this may take several minutes to complete")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to configure IAM-based access: {e}")
        return False

def create_opensearch_client(endpoint, region, service_account_role_arn=None):
    """Create OpenSearch client with AWS authentication"""
    try:
        # Parse endpoint to get host
        host = endpoint.replace('https://', '').replace('http://', '')
        
        # Get AWS credentials - either assume the service account role or use current credentials
        if service_account_role_arn:
            logger.info(f"Assuming role: {service_account_role_arn}")
            sts_client = boto3.client('sts', region_name=region)
            
            try:
                # Assume the service account role
                response = sts_client.assume_role(
                    RoleArn=service_account_role_arn,
                    RoleSessionName='opensearch-index-setup'
                )
                
                # Create session with assumed role credentials
                session = boto3.Session(
                    aws_access_key_id=response['Credentials']['AccessKeyId'],
                    aws_secret_access_key=response['Credentials']['SecretAccessKey'],
                    aws_session_token=response['Credentials']['SessionToken'],
                    region_name=region
                )
                credentials = session.get_credentials()
                logger.info("Successfully assumed service account role")
                
            except Exception as e:
                logger.warning(f"Failed to assume role {service_account_role_arn}: {e}")
                logger.info("Falling back to current credentials")
                session = boto3.Session()
                credentials = session.get_credentials()
        else:
            # Use current credentials
            session = boto3.Session()
            credentials = session.get_credentials()
        
        # Create AWS4Auth for authentication
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            'es',
            session_token=credentials.token
        )
        
        # Create OpenSearch client
        client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        
        return client
    except Exception as e:
        logger.error(f"Failed to create OpenSearch client: {e}")
        raise

def create_opensearch_index(endpoint, region, index_name, dimension=384, service_account_role_arn=None):
    """Create OpenSearch index with vector mapping"""
    max_retries = 3
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Connection attempt {attempt + 1}/{max_retries}")
            
            # Create client
            client = create_opensearch_client(endpoint, region, service_account_role_arn)
            
            # Test connection
            logger.info("Testing connection to OpenSearch...")
            cluster_info = client.info()
            logger.info(f"Connected to OpenSearch cluster: {cluster_info['cluster_name']}")
            
            # Check if index already exists
            if client.indices.exists(index=index_name):
                logger.info(f"Index '{index_name}' already exists")
                return True
            
            # Create index with vector mapping
            index_mapping = {
                "settings": {
                    "index": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                        "knn": True,
                        "knn.algo_param.ef_search": 100
                    }
                },
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 24
                                }
                            }
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "keyword"},
                                "chunk_id": {"type": "keyword"},
                                "timestamp": {"type": "date"}
                            }
                        }
                    }
                }
            }
            
            logger.info(f"Creating index '{index_name}' with {dimension}-dimensional vectors...")
            response = client.indices.create(index=index_name, body=index_mapping)
            logger.info(f"Index created successfully: {response}")
            
            return True
            
        except (ConnectionError, ConnectionTimeout) as e:
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay} seconds...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to create index after {max_retries} attempts")
                logger.error(f"Full traceback: {e}")
                raise e
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise e

def main():
    # Get configuration from environment
    endpoint = os.getenv('OPENSEARCH_ENDPOINT')
    region = os.getenv('AWS_REGION', 'us-east-1')
    index_name = os.getenv('VECTOR_INDEX_NAME', 'knowledge-embeddings')
    dimension = int(os.getenv('EMBEDDING_DIMENSION', '384'))
    service_account_role_arn = os.getenv('SERVICE_ACCOUNT_ROLE_ARN')
    
    if not endpoint:
        logger.error("OPENSEARCH_ENDPOINT environment variable is required")
        return False
    
    print(f"🔧 Setting up OpenSearch index...")
    print(f"   Endpoint: {endpoint}")
    print(f"   Region: {region}")
    print(f"   Index: {index_name}")
    print(f"   Dimension: {dimension}")
    if service_account_role_arn:
        print(f"   Using Role: {service_account_role_arn}")
        
        # Update the role trust policy to allow current role to assume it
        print("🔐 Updating role trust policy...")
        if update_role_trust_policy(service_account_role_arn):
            print("✅ Role trust policy updated successfully")
            # Wait a moment for the policy to propagate
            time.sleep(5)
        else:
            print("⚠️  Failed to update role trust policy, will try without role assumption")
            service_account_role_arn = None
            
        # Update OpenSearch domain access policy
        if service_account_role_arn:
            print("🔐 Updating OpenSearch domain access policy...")
            # Extract domain name from CloudFormation stack (it should be 'strandsdk-rag-opensearch')
            domain_name = 'strandsdk-rag-opensearch'
            if update_opensearch_access_policy(domain_name, region, service_account_role_arn):
                print("✅ OpenSearch domain access policy updated successfully")
                # Wait a moment for the policy to propagate
                time.sleep(5)
            else:
                print("⚠️  Failed to update OpenSearch domain access policy")
                
            # Configure fine-grained access control to use IAM instead of internal user database
            print("🔐 Configuring fine-grained access control for IAM...")
            if configure_opensearch_iam_access(domain_name, region, service_account_role_arn):
                print("✅ Fine-grained access control configured for IAM successfully")
                print("⏳ Waiting for domain configuration to update (this may take 10-15 minutes)...")
                # Wait longer for the domain configuration to update
                time.sleep(30)
            else:
                print("⚠️  Failed to configure fine-grained access control for IAM")
    print()
    
    try:
        success = create_opensearch_index(endpoint, region, index_name, dimension, service_account_role_arn)
        if success:
            print("✅ OpenSearch index created successfully!")
            print(f"   Index name: {index_name}")
            print(f"   Vector dimension: {dimension}")
            print(f"   Endpoint: {endpoint}")
            return True
        else:
            print("❌ Failed to create OpenSearch index")
            return False
    except Exception as e:
        print("❌ Failed to create OpenSearch index")
        print(f"   Check the logs above for details")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
