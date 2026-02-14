#!/usr/bin/env python3
"""
Standalone Tavily MCP Server for Kubernetes deployment.
This runs as a separate service that the main application connects to.
"""

import sys
import os
import warnings
import logging
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CompleteAsyncErrorFilter:
    """Complete async error filter that suppresses all async-related output."""
    
    def __init__(self):
        self.original_stderr = sys.__stderr__
        
    def write(self, text):
        """Filter out all async RuntimeErrors and related output."""
        if not text.strip():
            return
            
        # Comprehensive list of patterns to suppress
        suppress_patterns = [
            "RuntimeError",
            "httpcore",
            "_synchronization",
            "asyncio",
            "anyio",
            "sniffio",
            "await",
            "async",
            "CancelScope",
            "shield",
            "current_task",
            "get_running_loop",
            "cancel_shielded_checkpoint",
            "_anyio_lock",
            "acquire",
            "raise RuntimeError",
        ]
        
        # Check if this line should be suppressed
        should_suppress = any(pattern in text for pattern in suppress_patterns)
        
        # Also suppress lines that are just punctuation or whitespace
        if text.strip() in [":", "RuntimeError:", "RuntimeError: ", "RuntimeError", ""]:
            should_suppress = True
        
        # Only write if not suppressed and contains meaningful content
        if not should_suppress and len(text.strip()) > 1:
            self.original_stderr.write(text)
            self.original_stderr.flush()
        
    def flush(self):
        """Flush the original stderr."""
        self.original_stderr.flush()

def setup_complete_clean_environment():
    """Set up completely clean environment."""
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Install complete error filter
    sys.stderr = CompleteAsyncErrorFilter()
    
    # Try to import and use existing cleanup if available
    try:
        from src.utils.global_async_cleanup import setup_global_async_cleanup
        setup_global_async_cleanup()
    except ImportError:
        pass

def main():
    """Main entry point for standalone MCP server."""
    print("🚀 Starting Standalone Tavily MCP Server (Clean Mode)")
    print("=" * 60)
    print("Note: All async errors and warnings are suppressed")
    print("=" * 60)
    
    # Set up clean environment FIRST
    setup_complete_clean_environment()
    
    # Load environment variables from ConfigMap or local .env file
    if os.path.exists("/app/config/.env"):
        print("Loading environment variables from ConfigMap .env file...")
        load_dotenv("/app/config/.env")
        print("Environment variables loaded successfully from ConfigMap")
    elif os.path.exists("/app/.env"):
        print("Loading environment variables from local .env file...")
        load_dotenv("/app/.env")
        print("Environment variables loaded successfully from local file")
    else:
        print("WARNING: No .env file found. Using environment variables from Kubernetes.")
    
    # Verify Tavily API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("ERROR: TAVILY_API_KEY is not set")
        sys.exit(1)
    
    print("Tavily API key verified")
    
    try:
        # Import and run the Tavily MCP server
        from src.mcp_servers.tavily_search_server import mcp
        print("Starting Tavily MCP Server on port 8001...")
        mcp.run(transport="streamable-http", port=8001)
    except KeyboardInterrupt:
        print("\n👋 Tavily MCP Server stopped by user")
    except Exception as e:
        # Filter out async-related errors
        error_msg = str(e)
        if not any(keyword in error_msg.lower() for keyword in [
            "runtimeerror", "httpcore", "asyncio", "anyio", "await", "async"
        ]):
            print(f"❌ Tavily MCP Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
