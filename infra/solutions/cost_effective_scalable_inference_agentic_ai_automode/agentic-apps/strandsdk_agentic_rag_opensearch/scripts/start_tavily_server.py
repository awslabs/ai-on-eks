#!/usr/bin/env python3
"""
Startup script for Tavily MCP Server
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment variables from {env_file}")
else:
    print(f"⚠️  No .env file found at {env_file}")
    print("Please create a .env file with your configuration")

def check_tavily_api_key():
    """Check if Tavily API key is configured"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("❌ TAVILY_API_KEY environment variable is not set!")
        print("Please get your API key from https://tavily.com and set it in your .env file:")
        print("TAVILY_API_KEY=your-api-key-here")
        return False
    print(f"✅ Tavily API key configured: {api_key[:8]}...")
    return True

def check_server_health(max_retries=10, delay=2):
    """Check if the Tavily MCP server is running and healthy"""
    for attempt in range(max_retries):
        try:
            # Try to connect to the MCP server endpoint
            response = requests.get("http://localhost:8001/", timeout=5)
            if response.status_code in [200, 404]:  # 404 is OK for MCP server root
                print("✅ Tavily MCP server is healthy and ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if attempt < max_retries - 1:
            print(f"⏳ Waiting for server to start... (attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
    
    print("❌ Server health check failed after maximum retries")
    return False

def start_tavily_server():
    """Start the Tavily MCP server"""
    if not check_tavily_api_key():
        return False
    
    print("🚀 Starting Tavily MCP Server...")
    
    # Path to the server script
    server_script = project_root / "src" / "mcp_servers" / "tavily_search_server.py"
    
    if not server_script.exists():
        print(f"❌ Server script not found: {server_script}")
        return False
    
    try:
        # Start the server as a subprocess
        process = subprocess.Popen([
            sys.executable, str(server_script)
        ], cwd=str(project_root))
        
        print(f"📡 Server started with PID: {process.pid}")
        print("🔗 MCP server available at: http://localhost:8001/mcp")
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is healthy
        if check_server_health():
            print("\n🎉 Tavily MCP Server is ready!")
            print("\nAvailable tools:")
            print("  - web_search: General web search with AI-generated answers")
            print("  - news_search: Recent news and current events search")
            print("  - health_check: Service health status")
            print("\n💡 The supervisor agent will automatically use web search when RAG relevance is low (<0.3)")
            return True
        else:
            print("❌ Server failed to start properly")
            process.terminate()
            return False
            
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

if __name__ == "__main__":
    success = start_tavily_server()
    if success:
        print("\n✨ Server is running! Press Ctrl+C to stop.")
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
    else:
        print("\n❌ Failed to start Tavily MCP Server")
        sys.exit(1)
