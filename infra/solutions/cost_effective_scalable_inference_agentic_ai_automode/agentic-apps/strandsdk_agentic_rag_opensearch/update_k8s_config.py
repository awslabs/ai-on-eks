#!/usr/bin/env python3
"""
Interactive script to update Kubernetes ConfigMap and Secrets
for the Strands SDK RAG application with OpenSearch
"""

import os
import sys
import base64
import yaml
from typing import Dict, Any

def get_user_input(prompt: str, default: str = "", required: bool = True) -> str:
    """Get user input with optional default value"""
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "
    
    while True:
        value = input(full_prompt).strip()
        if value:
            return value
        elif default:
            return default
        elif not required:
            return ""
        else:
            print("This field is required. Please enter a value.")

def encode_base64(value: str) -> str:
    """Encode string to base64"""
    if not value:
        return "<BASE64_ENCODED_VALUE>"
    return base64.b64encode(value.encode('utf-8')).decode('utf-8')

def get_opensearch_info_from_deployment():
    """Try to extract OpenSearch info from previous deployment"""
    try:
        # Try to get from AWS CLI if available
        import subprocess
        result = subprocess.run([
            'aws', 'cloudformation', 'describe-stacks',
            '--stack-name', 'strandsdk-rag-opensearch-stack',
            '--region', 'us-east-1',
            '--query', 'Stacks[0].Outputs[?OutputKey==`OpenSearchDomainEndpoint`].OutputValue',
            '--output', 'text'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            endpoint = result.stdout.strip()
            return f"https://{endpoint}"
    except:
        pass
    
    return None

def main():
    print("🔧 Kubernetes ConfigMap and Secrets Update Tool")
    print("=" * 50)
    print("This script will help you update the k8s/configmap.yaml file")
    print("with your actual service endpoints and API keys.\n")
    
    # Get OpenSearch endpoint from deployment if possible
    opensearch_endpoint = get_opensearch_info_from_deployment()
    if opensearch_endpoint:
        print(f"✅ Found OpenSearch endpoint from deployment: {opensearch_endpoint}")
    
    # Collect configuration values
    print("📋 Configuration Values")
    print("-" * 25)
    
    config = {}
    
    # LiteLLM Configuration
    print("\n🤖 LiteLLM Configuration (for reasoning models):")
    config['litellm_base_url'] = get_user_input(
        "LiteLLM Base URL", 
        "http://your-litellm-loadbalancer.us-east-1.elb.amazonaws.com/v1"
    )
    config['reasoning_model'] = get_user_input(
        "Reasoning Model Name", 
        "vllm-server-qwen3"
    )
    
    # Embedding Configuration
    print("\n🔤 Embedding Configuration:")
    config['embedding_base_url'] = get_user_input(
        "Embedding Base URL", 
        config['litellm_base_url'].replace('/v1', '/v1/embeddings')
    )
    config['embedding_model'] = get_user_input(
        "Embedding Model Name", 
        "llamacpp-embedding"
    )
    
    # AWS Configuration
    print("\n☁️  AWS Configuration:")
    config['aws_region'] = get_user_input("AWS Region", "us-east-1")
    config['opensearch_endpoint'] = get_user_input(
        "OpenSearch Endpoint", 
        opensearch_endpoint or "https://your-opensearch-domain.us-east-1.es.amazonaws.com"
    )
    
    # Optional Langfuse Configuration
    print("\n📊 Langfuse Configuration (optional):")
    config['langfuse_host'] = get_user_input(
        "Langfuse Host URL", 
        "http://your-langfuse-loadbalancer.us-east-1.elb.amazonaws.com",
        required=False
    )
    
    # Application Settings
    print("\n⚙️  Application Settings:")
    config['vector_index_name'] = get_user_input("Vector Index Name", "knowledge-embeddings")
    config['top_k_results'] = get_user_input("Top K Results", "3")
    config['knowledge_dir'] = get_user_input("Knowledge Directory", "knowledge")
    config['output_dir'] = get_user_input("Output Directory", "output")
    
    # Collect secrets
    print("\n🔐 API Keys and Secrets")
    print("-" * 25)
    print("Note: Leave empty to keep placeholder values")
    
    secrets = {}
    
    secrets['litellm_api_key'] = get_user_input("LiteLLM API Key", required=False)
    secrets['embedding_api_key'] = get_user_input("Embedding API Key", required=False)
    secrets['tavily_api_key'] = get_user_input("Tavily API Key (for web search)", required=False)
    
    if config['langfuse_host']:
        secrets['langfuse_public_key'] = get_user_input("Langfuse Public Key", required=False)
        secrets['langfuse_secret_key'] = get_user_input("Langfuse Secret Key", required=False)
    else:
        secrets['langfuse_public_key'] = ""
        secrets['langfuse_secret_key'] = ""
    
    # Generate the updated ConfigMap
    print("\n🔄 Generating updated ConfigMap...")
    
    # Create the .env content
    env_content = f"""# LiteLLM Configuration for Reasoning Models
LITELLM_BASE_URL={config['litellm_base_url']}
REASONING_MODEL={config['reasoning_model']}

# Embedding Configuration (separate from reasoning)
EMBEDDING_BASE_URL={config['embedding_base_url']}
EMBEDDING_MODEL={config['embedding_model']}

# AWS Configuration  
AWS_REGION={config['aws_region']}
OPENSEARCH_ENDPOINT={config['opensearch_endpoint']}

# Tavily MCP Service Configuration
TAVILY_MCP_SERVICE_URL=http://tavily-mcp-service:8001/mcp

# Optional: Langfuse for observability
LANGFUSE_HOST={config['langfuse_host']}

# Application Settings
KNOWLEDGE_DIR={config['knowledge_dir']}
OUTPUT_DIR={config['output_dir']}
VECTOR_INDEX_NAME={config['vector_index_name']}
TOP_K_RESULTS={config['top_k_results']}"""
    
    # Create the full YAML content as a string to preserve formatting
    yaml_content = f"""apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  labels:
    app: strandsdk-rag
data:
  .env: |
{env_content}
  
  # Individual config values for environment variables
  litellm-base-url: "{config['litellm_base_url']}"
  reasoning-model: "{config['reasoning_model']}"
  embedding-base-url: "{config['embedding_base_url']}"
  embedding-model: "{config['embedding_model']}"
  opensearch-endpoint: "{config['opensearch_endpoint']}"
  aws-region: "{config['aws_region']}"
  vector-index-name: "{config['vector_index_name']}"
  tavily-mcp-service-url: "http://tavily-mcp-service:8001/mcp"
  langfuse-host: "{config['langfuse_host']}"
  knowledge-dir: "{config['knowledge_dir']}"
  output-dir: "{config['output_dir']}"
  top-k-results: "{config['top_k_results']}"
---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  labels:
    app: strandsdk-rag
type: Opaque
data:
  # Base64 encoded secrets - replace with your actual base64 encoded values
  # To encode: echo -n "your-api-key" | base64
  litellm-api-key: {encode_base64(secrets['litellm_api_key'])}
  embedding-api-key: {encode_base64(secrets['embedding_api_key'])}
  tavily-api-key: {encode_base64(secrets['tavily_api_key'])}
  langfuse-public-key: {encode_base64(secrets['langfuse_public_key'])}
  langfuse-secret-key: {encode_base64(secrets['langfuse_secret_key'])}"""
    
    # Write the updated ConfigMap file
    output_file = 'k8s/configmap.yaml'
    backup_file = 'k8s/configmap.yaml.backup'
    
    # Create backup
    if os.path.exists(output_file):
        print(f"📁 Creating backup: {backup_file}")
        import shutil
        shutil.copy2(output_file, backup_file)
    
    # Write new file
    with open(output_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Updated ConfigMap written to: {output_file}")
    
    # Show summary
    print("\n📋 Configuration Summary:")
    print("-" * 30)
    print(f"LiteLLM Base URL: {config['litellm_base_url']}")
    print(f"Reasoning Model: {config['reasoning_model']}")
    print(f"Embedding Base URL: {config['embedding_base_url']}")
    print(f"Embedding Model: {config['embedding_model']}")
    print(f"OpenSearch Endpoint: {config['opensearch_endpoint']}")
    print(f"AWS Region: {config['aws_region']}")
    print(f"Vector Index: {config['vector_index_name']}")
    if config['langfuse_host']:
        print(f"Langfuse Host: {config['langfuse_host']}")
    
    secrets_count = len([k for k, v in secrets.items() if v])
    print(f"\n🔐 Secrets configured: {secrets_count}/{len(secrets)}")
    
    if secrets_count == 0:
        print("⚠️  No API keys provided - placeholder values will be used")
        print("   You can update secrets later with: kubectl edit secret app-secrets")
    
    print("\n🚀 Next Steps:")
    print("1. Review the generated k8s/configmap.yaml file")
    print("2. Apply the ConfigMap to your cluster:")
    print("   kubectl apply -f k8s/configmap.yaml")
    print("3. Deploy your application:")
    print("   kubectl apply -f k8s/")
    
    # Show how to encode secrets manually if needed
    if secrets_count < len(secrets):
        print("\n💡 To encode API keys manually:")
        print("   echo -n 'your-api-key' | base64")
        print("   Then update the secret with: kubectl edit secret app-secrets")
    
    # Offer to apply directly
    apply_now = input("\n❓ Would you like to apply the ConfigMap now? (y/N): ").strip().lower()
    if apply_now in ['y', 'yes']:
        try:
            import subprocess
            result = subprocess.run(['kubectl', 'apply', '-f', output_file], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ ConfigMap applied successfully!")
                print(result.stdout)
            else:
                print("❌ Failed to apply ConfigMap:")
                print(result.stderr)
        except Exception as e:
            print(f"❌ Error applying ConfigMap: {e}")
            print("Please apply manually with: kubectl apply -f k8s/configmap.yaml")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
