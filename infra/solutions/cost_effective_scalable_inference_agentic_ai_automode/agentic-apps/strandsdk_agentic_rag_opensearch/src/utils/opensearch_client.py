"""OpenSearch client wrapper for the multi-agent RAG system."""

import logging
from typing import Optional, Dict, Any
from ..tools.opensearch_vector_store import OpenSearchVectorStore

logger = logging.getLogger(__name__)

class OpenSearchClient:
    """OpenSearch client wrapper that provides compatibility with server.py expectations."""
    
    def __init__(self, config):
        """Initialize OpenSearch client with configuration."""
        self.config = config
        self._vector_store = None
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenSearch client using the existing vector store."""
        try:
            # Use the existing OpenSearchVectorStore implementation
            self._vector_store = OpenSearchVectorStore()
            
            # Expose the underlying client for compatibility
            if self._vector_store and self._vector_store.client:
                self.client = self._vector_store.client
                logger.info("OpenSearch client initialized successfully via vector store")
            else:
                logger.warning("OpenSearch vector store client not available")
                self.client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Check if the client is connected and can reach OpenSearch."""
        if not self.client:
            return False
        
        try:
            info = self.client.info()
            return True
        except Exception as e:
            logger.debug(f"OpenSearch connection check failed: {e}")
            return False
    
    def get_info(self) -> Optional[Dict[str, Any]]:
        """Get OpenSearch cluster information."""
        if not self.client:
            return None
        
        try:
            return self.client.info()
        except Exception as e:
            logger.error(f"Failed to get OpenSearch info: {e}")
            return None
    
    def index_exists(self, index_name: str) -> bool:
        """Check if an index exists."""
        if not self.client:
            return False
        
        try:
            return self.client.indices.exists(index=index_name)
        except Exception as e:
            logger.error(f"Failed to check if index exists: {e}")
            return False
    
    def get_document_count(self, index_name: str) -> int:
        """Get the number of documents in an index."""
        if not self.client:
            return 0
        
        try:
            response = self.client.count(index=index_name)
            return response.get("count", 0)
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def close(self) -> None:
        """Close the OpenSearch connection."""
        if self._vector_store:
            try:
                self._vector_store.close()
                logger.info("OpenSearch connection closed")
            except Exception as e:
                logger.error(f"Error closing OpenSearch connection: {e}")
