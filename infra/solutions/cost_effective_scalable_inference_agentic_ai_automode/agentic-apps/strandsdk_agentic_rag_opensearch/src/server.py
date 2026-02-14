#!/usr/bin/env python3
"""
FastAPI Server for Multi-Agent RAG System with MCP and OpenSearch using Strands SDK

A REST API server that provides HTTP endpoints for the multi-agent system.
Uses clean execution pattern to suppress async warnings.
"""

import sys
import os
import warnings
import logging
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

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
            "File \"/home/ubuntu/Cost_Effective_and_Scalable_Models_Inference_on_AWS_Graviton/agentic-apps/strandsdk_agentic_rag_opensearch/venv/lib/python3.10/site-packages/httpcore",
            "File \"/usr/lib/python3.10/asyncio",
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

# Set up clean environment FIRST before any other imports
setup_complete_clean_environment()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.config import config
from src.utils.logging import setup_logging, log_title
from src.agents.supervisor_agent import supervisor_agent, create_fresh_supervisor_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.mcp_agent import mcp_agent

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the multi-agent system", max_length=1000)
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The response from the multi-agent system")
    session_id: Optional[str] = Field(None, description="Session ID if provided")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(default="success", description="Response status")

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Health status")
    version: str = Field(default="1.0.0", description="Application version")
    services: Dict[str, str] = Field(..., description="Status of various services")

class EmbedRequest(BaseModel):
    force_refresh: bool = Field(default=False, description="Force refresh of all embeddings")

# Global variables for service status
tavily_server_process = None
service_status = {
    "tavily_mcp_server": "starting",
    "opensearch": "unknown",
    "knowledge_base": "unknown"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global tavily_server_process, service_status
    
    # Startup
    logger = logging.getLogger(__name__)
    log_title("FASTAPI SERVER STARTUP (CLEAN MODE)")
    
    try:
        # Validate configuration
        config.validate_config()
        logger.info("Configuration validated successfully")
        
        # Check Tavily MCP Server connectivity (Kubernetes service)
        logger.info("Checking Tavily MCP Server connectivity...")
        tavily_accessible = await check_tavily_server()
        if tavily_accessible:
            service_status["tavily_mcp_server"] = "connected"
            logger.info("Tavily MCP Server is accessible via Kubernetes service")
        else:
            service_status["tavily_mcp_server"] = "disconnected"
            logger.warning("Tavily MCP Server is not accessible")
        
        # Check OpenSearch connectivity
        try:
            from src.utils.opensearch_client import OpenSearchClient
            client = OpenSearchClient(config)
            # Simple connectivity test
            info = client.client.info()
            service_status["opensearch"] = "connected"
            logger.info(f"OpenSearch connected: {info.get('version', {}).get('number', 'unknown')}")
        except Exception as e:
            service_status["opensearch"] = "disconnected"
            logger.warning(f"OpenSearch connection failed: {e}")
        
        # Check knowledge base status
        try:
            from src.utils.opensearch_client import OpenSearchClient
            client = OpenSearchClient(config)
            if client.client.indices.exists(index=config.VECTOR_INDEX_NAME):
                count = client.client.count(index=config.VECTOR_INDEX_NAME)
                doc_count = count['count']
                service_status["knowledge_base"] = f"ready ({doc_count} documents)"
                logger.info(f"Knowledge base ready with {doc_count} documents")
            else:
                service_status["knowledge_base"] = "no_index"
                logger.warning("Knowledge base index does not exist")
        except Exception as e:
            service_status["knowledge_base"] = "error"
            logger.warning(f"Knowledge base check failed: {e}")
        
        # Initialize MCP client during startup to avoid blocking during requests
        try:
            from src.agents.supervisor_agent import get_tavily_mcp_client
            mcp_client = get_tavily_mcp_client()
            if mcp_client:
                # Test MCP client connectivity
                with mcp_client:
                    tools = mcp_client.list_tools_sync()
                    service_status["mcp_tools"] = f"ready ({len(tools)} tools)"
                    logger.info(f"MCP client initialized successfully with {len(tools)} tools")
            else:
                service_status["mcp_tools"] = "unavailable"
                logger.warning("MCP client initialization failed")
        except Exception as e:
            service_status["mcp_tools"] = "error"
            logger.warning(f"MCP client initialization failed: {e}")
        
        logger.info("FastAPI server startup completed")
        
        yield
        
    except Exception as e:
        # Filter out async-related errors in startup
        error_msg = str(e)
        if not any(keyword in error_msg.lower() for keyword in [
            "runtimeerror", "httpcore", "asyncio", "anyio", "await", "async"
        ]):
            logger.error(f"Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down FastAPI server...")
    # No need to terminate Tavily server as it's running in a separate Kubernetes service

async def check_tavily_server():
    """Check if the Tavily MCP server is accessible via Kubernetes service."""
    try:
        import httpx
        
        # Get Tavily service URL from environment or use default Kubernetes service name
        tavily_service_url = os.getenv("TAVILY_MCP_SERVICE_URL", "http://tavily-mcp-service:8001")
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Try to connect to the MCP endpoint with proper headers
            response = await client.post(
                f"{tavily_service_url}/mcp/",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                },
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "id": "health-check",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "health-check",
                            "version": "1.0.0"
                        }
                    }
                }
            )
            # If we get any response (even an error), the service is accessible
            return response.status_code in [200, 400, 406]  # Accept various response codes as "accessible"
    except Exception as e:
        # Filter out async-related errors
        error_msg = str(e)
        if not any(keyword in error_msg.lower() for keyword in [
            "runtimeerror", "httpcore", "asyncio", "anyio", "await", "async"
        ]):
            logging.error(f"Failed to check Tavily server: {e}")
        return False

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent RAG System API",
    description="REST API for the Strands SDK Multi-Agent RAG System with OpenSearch (Clean Mode)",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        services=service_status
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multi-Agent RAG System API (Clean Mode)",
        "version": "1.0.0",
        "mode": "clean",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "embed": "/embed",
            "status": "/status",
            "docs": "/docs"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query using the multi-agent system."""
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question[:50]}...")
        
        # Validate query length
        if len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Limit query length to avoid context window issues
        query = request.question
        if len(query) > 1000:
            logger.warning("Query too long, truncating to 1000 characters")
            query = query[:1000]
        
        # Create a fresh agent instance for each query to avoid context accumulation
        fresh_agent = create_fresh_supervisor_agent()
        
        # Process the query
        response = fresh_agent(query)
        
        # Ensure response is properly formatted
        if response is None:
            response = "No response received from agent."
        
        response_str = str(response).strip()
        if not response_str:
            response_str = "Agent completed processing but returned empty response."
        
        # Limit response length if needed
        if len(response_str) > 4000:
            logger.warning("Response too long, truncating to 4000 characters")
            response_str = response_str[:4000] + "... [Response truncated due to length]"
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed successfully in {processing_time:.2f}s")
        
        return QueryResponse(
            response=response_str,
            session_id=request.session_id,
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        # Always log the error for debugging, but filter display for async-related errors
        error_msg = str(e)
        logger.error(f"Error processing query: {e}", exc_info=True)
        
        # Filter display message for async-related errors
        if any(keyword in error_msg.lower() for keyword in [
            "runtimeerror", "httpcore", "asyncio", "anyio", "await", "async"
        ]):
            display_error = "Internal processing error (async-related)"
        else:
            display_error = str(e) if str(e) else "Unknown error occurred"
        
        return QueryResponse(
            response=f"Error processing query: {display_error}",
            session_id=request.session_id,
            processing_time=processing_time,
            status="error"
        )

@app.post("/embed")
async def embed_knowledge(request: EmbedRequest, background_tasks: BackgroundTasks):
    """Embed knowledge documents into the vector database."""
    try:
        logger.info("Starting knowledge embedding process...")
        
        # Run embedding in background to avoid timeout
        def run_embedding():
            try:
                if request.force_refresh:
                    result = knowledge_agent('Please refresh and embed all knowledge files')
                else:
                    result = knowledge_agent('Please embed all knowledge files')
                
                # Update service status
                global service_status
                try:
                    from src.utils.opensearch_client import OpenSearchClient
                    client = OpenSearchClient(config)
                    if client.client.indices.exists(index=config.VECTOR_INDEX_NAME):
                        count = client.client.count(index=config.VECTOR_INDEX_NAME)
                        doc_count = count['count']
                        service_status["knowledge_base"] = f"ready ({doc_count} documents)"
                except Exception:
                    pass
                
                logger.info("Knowledge embedding completed successfully")
                return result
            except Exception as e:
                # Filter out async-related errors
                error_msg = str(e)
                if not any(keyword in error_msg.lower() for keyword in [
                    "runtimeerror", "httpcore", "asyncio", "anyio", "await", "async"
                ]):
                    logger.error(f"Knowledge embedding failed: {e}")
                service_status["knowledge_base"] = "error"
                raise
        
        background_tasks.add_task(run_embedding)
        
        return {
            "message": "Knowledge embedding started in background",
            "status": "processing"
        }
        
    except Exception as e:
        # Filter out async-related errors
        error_msg = str(e)
        if not any(keyword in error_msg.lower() for keyword in [
            "runtimeerror", "httpcore", "asyncio", "anyio", "await", "async"
        ]):
            logger.error(f"Error starting embedding process: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start embedding: {str(e)}")

@app.get("/status")
async def get_status():
    """Get detailed system status."""
    return {
        "mode": "clean",
        "services": service_status,
        "config": {
            "opensearch_endpoint": config.OPENSEARCH_ENDPOINT,
            "knowledge_dir": config.KNOWLEDGE_DIR,
            "vector_index": config.VECTOR_INDEX_NAME,
            "reasoning_model": config.REASONING_MODEL,
            "embedding_model": config.EMBEDDING_MODEL
        }
    }

def run_server():
    """Run the FastAPI server."""
    print("🚀 Starting Multi-Agent RAG System API Server (Clean Mode)")
    print("=" * 60)
    print("Note: All async errors and warnings are suppressed")
    print("=" * 60)
    
    # Load environment variables from ConfigMap or local file
    if os.path.exists("/app/config/.env"):
        print("Loading environment variables from ConfigMap...")
        from dotenv import load_dotenv
        load_dotenv("/app/config/.env")
        print("Environment variables loaded from ConfigMap")
    elif os.path.exists("/app/.env"):
        print("Loading environment variables from local file...")
        from dotenv import load_dotenv
        load_dotenv("/app/.env")
        print("Environment variables loaded from local file")
    
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        timeout_keep_alive=900,  # 15 minutes keep-alive timeout
        timeout_graceful_shutdown=30,  # 30 seconds graceful shutdown
        limit_max_requests=1000,  # Limit max requests per worker
        limit_concurrency=100  # Limit concurrent connections
    )

if __name__ == "__main__":
    run_server()
