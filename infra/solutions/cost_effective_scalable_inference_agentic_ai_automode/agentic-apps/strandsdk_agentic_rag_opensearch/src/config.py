"""Configuration management for the multi-agent RAG system."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # LiteLLM Configuration for Reasoning Models
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LITELLM_BASE_URL: str = os.getenv("LITELLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    REASONING_MODEL: str = os.getenv("REASONING_MODEL", "qwen-qwq-32b-preview")
    
    # Embedding Configuration (separate from reasoning)
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "llamacpp-embedding")
    
    # Legacy OpenAI Configuration (for backward compatibility)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", os.getenv("REASONING_MODEL", "qwen-qwq-32b-preview"))
    
    # AWS Configuration
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    OPENSEARCH_ENDPOINT: str = os.getenv("OPENSEARCH_ENDPOINT", "")
    
    # Tavily MCP Configuration
    TAVILY_MCP_SERVICE_URL: str = os.getenv("TAVILY_MCP_SERVICE_URL", "http://localhost:8001/mcp")
    
    # Langfuse Configuration
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "")
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    
    # Application Configuration
    KNOWLEDGE_DIR: str = os.getenv("KNOWLEDGE_DIR", "knowledge")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
    EMBEDDING_ENDPOINT: str = os.getenv("EMBEDDING_ENDPOINT", "")
    
    # Vector Search Configuration
    VECTOR_INDEX_NAME: str = os.getenv("VECTOR_INDEX_NAME", "knowledge-embeddings")
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    @classmethod
    def is_langfuse_enabled(cls) -> bool:
        """Check if Langfuse is properly configured."""
        return bool(cls.LANGFUSE_HOST and cls.LANGFUSE_PUBLIC_KEY and cls.LANGFUSE_SECRET_KEY)
    
    @classmethod
    def validate_config(cls) -> None:
        """Validate required configuration."""
        required_vars = [
            ("LITELLM_API_KEY", cls.LITELLM_API_KEY),
            ("OPENSEARCH_ENDPOINT", cls.OPENSEARCH_ENDPOINT),
        ]
        
        missing_vars = [name for name, value in required_vars if not value]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Global config instance
config = Config()
