#!/bin/bash
set -e

echo "Starting Strands SDK Agentic RAG Main Application (Clean Mode)..."

# Load environment variables from ConfigMap or local file
if [ -f "/app/config/.env" ]; then
    echo "Loading environment variables from ConfigMap .env file..."
    export $(grep -v '^#' /app/config/.env | xargs)
    echo "Environment variables loaded successfully from ConfigMap"
elif [ -f "/app/.env" ]; then
    echo "Loading environment variables from local .env file..."
    export $(grep -v '^#' /app/.env | xargs)
    echo "Environment variables loaded successfully from local file"
else
    echo "WARNING: No .env file found. Using environment variables from Kubernetes."
fi

# Verify critical environment variables
echo "Verifying critical environment variables..."
if [ -z "$LITELLM_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Neither LITELLM_API_KEY nor OPENAI_API_KEY is set"
    exit 1
fi

if [ -z "$OPENSEARCH_ENDPOINT" ]; then
    echo "ERROR: OPENSEARCH_ENDPOINT is not set"
    exit 1
fi

if [ -z "$AWS_REGION" ]; then
    echo "ERROR: AWS_REGION is not set"
    exit 1
fi

echo "Critical environment variables verified"

echo "Starting FastAPI server with clean mode..."
echo "Server will be available on port 8000"
echo "API Documentation available at http://localhost:8000/docs"

# Load .env and run the server
python -c "
from dotenv import load_dotenv
import os

# Try to load from ConfigMap first, then fallback to local
if os.path.exists('/app/config/.env'):
    load_dotenv('/app/config/.env')
    print('Loaded environment from ConfigMap')
elif os.path.exists('/app/.env'):
    load_dotenv('/app/.env')
    print('Loaded environment from local file')

# Now run the server
from src.server import run_server
run_server()
"
