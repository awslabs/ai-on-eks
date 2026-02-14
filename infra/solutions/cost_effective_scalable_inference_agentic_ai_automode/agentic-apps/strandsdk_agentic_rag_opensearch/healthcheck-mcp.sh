#!/bin/bash

# Health check script for MCP Server

# Load environment variables from ConfigMap or local file
if [ -f "/app/config/.env" ]; then
    export $(grep -v '^#' /app/config/.env | xargs) 2>/dev/null || true
elif [ -f "/app/.env" ]; then
    export $(grep -v '^#' /app/.env | xargs) 2>/dev/null || true
fi

# Check if the MCP server is responding
if curl -f -s http://localhost:8001/mcp/ > /dev/null 2>&1; then
    echo "MCP Server: Running"
    exit 0
else
    echo "MCP Server: Not responding"
    exit 1
fi
