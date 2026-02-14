#!/bin/bash

# Health check script for Main Application (FastAPI Server)

# Load environment variables from ConfigMap or local file
if [ -f "/app/config/.env" ]; then
    export $(grep -v '^#' /app/config/.env | xargs) 2>/dev/null || true
elif [ -f "/app/.env" ]; then
    export $(grep -v '^#' /app/.env | xargs) 2>/dev/null || true
fi

# Check if the FastAPI server is responding
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "FastAPI Server: Running"
    exit 0
else
    echo "FastAPI Server: Not responding"
    exit 1
fi
