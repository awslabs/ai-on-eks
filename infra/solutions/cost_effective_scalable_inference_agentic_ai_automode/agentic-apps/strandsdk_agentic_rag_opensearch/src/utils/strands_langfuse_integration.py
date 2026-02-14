"""
Strands SDK tracing integration with Langfuse support.

This module provides tracing capabilities using Strands SDK's native OpenTelemetry
integration, with optional Langfuse export for enhanced observability.
"""

import logging
import os
from typing import Dict, Any, Optional
from ..config import config

try:
    from strands.telemetry.tracer import get_tracer
    STRANDS_TRACING_AVAILABLE = True
except ImportError:
    STRANDS_TRACING_AVAILABLE = False
    get_tracer = None

logger = logging.getLogger(__name__)

class StrandsTracingIntegration:
    """Strands SDK tracing integration with Langfuse support."""
    
    def __init__(self):
        self.tracer = None
        self.tracing_enabled = False
        self._initialize_tracing()
    
    def _initialize_tracing(self) -> None:
        """Initialize Strands tracing with optional Langfuse export."""
        if not STRANDS_TRACING_AVAILABLE:
            logger.warning("Strands tracing not available. Install latest Strands SDK.")
            return
        
        try:
            # Configure tracing based on environment or config
            otlp_endpoint = None
            otlp_headers = {}
            enable_console = False
            
            # Check if Langfuse is configured and set up OTLP endpoint
            if config.is_langfuse_enabled():
                # For Langfuse, we can use their OTLP endpoint if available
                # Or set up console export for development
                enable_console = True
                logger.info("Langfuse configured - enabling console tracing for development")
            
            # Check for explicit OTLP configuration
            if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
                otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
                
            if os.getenv("OTEL_EXPORTER_OTLP_HEADERS"):
                headers_str = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
                otlp_headers = dict(h.split("=") for h in headers_str.split(","))
            
            if os.getenv("STRANDS_OTEL_ENABLE_CONSOLE_EXPORT", "").lower() == "true":
                enable_console = True
            
            # Initialize the tracer
            self.tracer = get_tracer(
                service_name="strands-agentic-rag",
                otlp_endpoint=otlp_endpoint,
                otlp_headers=otlp_headers,
                enable_console_export=enable_console
            )
            
            self.tracing_enabled = True
            logger.info("Strands tracing initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Strands tracing: {e}")
            self.tracer = None
            self.tracing_enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.tracing_enabled
    
    def create_agent_with_tracing(self, agent_class, trace_attributes: Optional[Dict[str, Any]] = None, **agent_kwargs):
        """
        Create a Strands Agent with built-in tracing enabled.
        
        Args:
            agent_class: Strands Agent class
            trace_attributes: Custom attributes to add to traces
            **agent_kwargs: Arguments for agent initialization
            
        Returns:
            Agent instance with tracing enabled
        """
        # Add trace attributes if provided
        if trace_attributes and self.tracing_enabled:
            agent_kwargs["trace_attributes"] = trace_attributes
        
        # Create and return the agent (tracing is automatic in Strands SDK)
        return agent_class(**agent_kwargs)
    
    def setup_environment_for_tracing(self):
        """Set up environment variables for optimal tracing."""
        if not self.tracing_enabled:
            return
        
        # Enable console export for development if not already set
        if not os.getenv("STRANDS_OTEL_ENABLE_CONSOLE_EXPORT"):
            os.environ["STRANDS_OTEL_ENABLE_CONSOLE_EXPORT"] = "true"
        
        # Set sampling if not configured (sample 100% for development)
        if not os.getenv("OTEL_TRACES_SAMPLER"):
            os.environ["OTEL_TRACES_SAMPLER"] = "always_on"
        
        logger.info("Tracing environment configured")

# Global instance
strands_tracing = StrandsTracingIntegration()

# Convenience functions
def create_traced_agent(agent_class, session_id: Optional[str] = None, 
                       user_id: Optional[str] = None, **kwargs):
    """Create a Strands Agent with tracing enabled."""
    trace_attributes = {}
    
    if session_id:
        trace_attributes["session.id"] = session_id
    if user_id:
        trace_attributes["user.id"] = user_id
    
    # Add default attributes
    trace_attributes.update({
        "service.name": "agentic-rag-opensearch",
        "service.version": "1.0.0",
        "tags": ["RAG", "OpenSearch", "Multi-Agent"]
    })
    
    return strands_tracing.create_agent_with_tracing(
        agent_class, 
        trace_attributes=trace_attributes,
        **kwargs
    )

def setup_tracing_environment():
    """Set up the tracing environment."""
    strands_tracing.setup_environment_for_tracing()

# Export main components
__all__ = [
    "StrandsTracingIntegration",
    "strands_tracing", 
    "create_traced_agent",
    "setup_tracing_environment"
]
