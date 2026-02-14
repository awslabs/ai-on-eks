from datetime import datetime
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from aws_requests_auth.aws_auth import AWSRequestsAuth
import boto3
from ..config import config

logger = logging.getLogger(__name__)

class OpenSearchVectorStore:
    """Vector store implementation using OpenSearch."""
    
    def __init__(self, index_name: str = None):
        self.index_name = index_name or config.VECTOR_INDEX_NAME
        self.client: Optional[OpenSearch] = None
        self.dimension = 384  # Default dimension for embeddings
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenSearch client with AWS authentication."""
        try:
            # Get AWS credentials
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if not credentials:
                raise ValueError("AWS credentials not found")
            
            # Parse endpoint to get host
            endpoint_url = config.OPENSEARCH_ENDPOINT
            if endpoint_url.startswith('https://'):
                host = endpoint_url.replace('https://', '')
            else:
                host = endpoint_url
            
            # Create AWS auth
            awsauth = AWSRequestsAuth(
                aws_access_key=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_token=credentials.token,
                aws_host=host,
                aws_region=config.AWS_REGION,
                aws_service='es'
            )
            
            # Initialize OpenSearch client
            self.client = OpenSearch(
                hosts=[{'host': host, 'port': 443}],
                http_auth=awsauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
            
            logger.info("OpenSearch client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch client: {e}")
            raise
    
    def create_index(self, dimension: int = 384) -> bool:
        """Create the vector index if it doesn't exist."""
        if not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        self.dimension = dimension
        
        try:
            # Check if index exists
            if self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} already exists")
                return True
            
            # Create index with vector mapping
            index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "knn.space_type": "cosinesimil"
                    }
                },
                "mappings": {
                    "properties": {
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "method": {
                                "name": "hnsw",
                                "space_type": "cosinesimil",
                                "engine": "nmslib",
                                "parameters": {
                                    "ef_construction": 128,
                                    "m": 16
                                }
                            }
                        },
                        "document": {
                            "type": "text",
                            "store": True
                        },
                        "metadata": {
                            "type": "object"
                        },
                        "timestamp": {
                            "type": "date"
                        }
                    }
                }
            }
            
            response = self.client.indices.create(
                index=self.index_name,
                body=index_body
            )
            
            logger.info(f"Created index {self.index_name}: {response}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False
    
    def add_embedding(self, embedding: List[float], document: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a single document with embedding to the vector store."""
        if not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        try:
            from datetime import datetime
            
            doc_body = {
                "embedding": embedding,
                "document": document,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            response = self.client.index(
                index=self.index_name,
                body=doc_body,
                refresh=True  # Make immediately searchable
            )
            
            logger.debug(f"Document added to OpenSearch: {response['_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return False
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add multiple documents with embeddings to the vector store."""
        if not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        try:
            # Prepare bulk operations
            bulk_body = []
            
            for doc in documents:
                # Index operation
                bulk_body.append({
                    "index": {
                        "_index": self.index_name,
                        "_id": doc.get("id", None)
                    }
                })
                
                # Document data
                bulk_body.append({
                    "embedding": doc["vector"],
                    "document": doc["content"],
                    "metadata": doc.get("metadata", {}),
                    "timestamp": doc.get("timestamp", datetime.now().isoformat())
                })
            
            # Execute bulk operation
            response = self.client.bulk(body=bulk_body, refresh=True)
            
            # Check for errors
            if response.get("errors"):
                logger.error(f"Bulk indexing errors: {response}")
                return False
            
            logger.info(f"Successfully indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        """Search for similar documents using vector similarity."""
        if not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        try:
            search_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                },
                "_source": ["document"]
            }
            
            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Extract documents from search results
            hits = response["hits"]["hits"]
            documents = [hit["_source"]["document"] for hit in hits]
            
            logger.debug(f"Found {len(documents)} similar documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            return []
    
    def similarity_search(
        self, 
        query_vector: List[float], 
        k: int = None, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform similarity search using vector with detailed results."""
        if not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        k = k or config.TOP_K_RESULTS
        
        try:
            # Build query with source filtering to reduce response size
            query = {
                "size": k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_vector,
                            "k": k
                        }
                    }
                },
                "_source": ["document", "metadata"]  # Only return necessary fields
            }
            
            # Add filters if provided
            if filter_dict:
                query["query"] = {
                    "bool": {
                        "must": [query["query"]],
                        "filter": [
                            {"term": {key: value}} for key, value in filter_dict.items()
                        ]
                    }
                }
            
            # Execute search
            response = self.client.search(
                index=self.index_name,
                body=query
            )
            
            # Process results - keep metadata minimal
            results = []
            for hit in response["hits"]["hits"]:
                # Extract only essential metadata to reduce token usage
                metadata = {}
                if "metadata" in hit["_source"]:
                    source_metadata = hit["_source"]["metadata"]
                    # Only keep essential metadata fields
                    if isinstance(source_metadata, dict):
                        metadata = {
                            "source": source_metadata.get("source", "Unknown")
                        }
                
                results.append({
                    "content": hit["_source"]["document"],
                    "metadata": metadata,
                    "score": hit["_score"],
                    "id": hit["_id"]
                })
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def delete_index(self) -> bool:
        """Delete the vector index."""
        if not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        try:
            if self.client.indices.exists(index=self.index_name):
                response = self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted index {self.index_name}: {response}")
                return True
            else:
                logger.info(f"Index {self.index_name} does not exist")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete index: {e}")
            return False
    
    def get_document_count(self) -> int:
        """Get the number of documents in the index."""
        if not self.client:
            raise RuntimeError("OpenSearch client not initialized")
        
        try:
            response = self.client.count(index=self.index_name)
            return response["count"]
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def close(self) -> None:
        """Close the OpenSearch connection."""
        if self.client:
            try:
                # OpenSearch client doesn't have explicit close method
                # Connection will be closed automatically
                logger.info("OpenSearch connection closed")
            except Exception as e:
                logger.error(f"Error closing OpenSearch connection: {e}")
