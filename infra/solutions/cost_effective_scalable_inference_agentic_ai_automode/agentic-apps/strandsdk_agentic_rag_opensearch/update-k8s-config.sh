#!/bin/bash

# Interactive Kubernetes ConfigMap Update Script
# Updates k8s/configmap.yaml with your actual service endpoints and API keys

echo "🚀 Starting Kubernetes ConfigMap Update Tool..."
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if k8s directory exists
if [ ! -d "k8s" ]; then
    echo "❌ k8s directory not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if configmap.yaml exists
if [ ! -f "k8s/configmap.yaml" ]; then
    echo "❌ k8s/configmap.yaml not found"
    echo "Please ensure the ConfigMap file exists"
    exit 1
fi

# Run the Python script
python3 ./update_k8s_config.py

echo ""
echo "✅ ConfigMap update completed!"
