"""Embedding retriever for RAG functionality."""

import logging
import math
import random
from typing import List, Dict, Any, Optional
import requests
from .opensearch_vector_store import OpenSearchVectorStore
from ..config import config
from ..utils.logging import log_title

logger = logging.getLogger(__name__)

class EmbeddingRetriever:
    """Handles embedding generation and retrieval operations."""
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.vector_store = OpenSearchVectorStore()
        self.embedding_endpoint = config.EMBEDDING_BASE_URL
        self.api_key = config.EMBEDDING_API_KEY
        self.target_dimension = 384  # Target dimension for embeddings
    
    def embed_document(self, document: str) -> List[float]:
        """Embed a document and add it to the vector store."""
        log_title('EMBEDDING DOCUMENT')
        embedding = self.embed(document)
        self.vector_store.add_embedding(embedding, document)
        return embedding
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query for search."""
        log_title('EMBEDDING QUERY')
        return self.embed(query)
    
    def generate_random_embedding(self) -> List[float]:
        """Generate a random embedding as fallback."""
        return [random.uniform(-1, 1) for _ in range(self.target_dimension)]
    
    def normalize_vector(self, vector: List[float]) -> List[float]:
        """Normalize a vector to unit length."""
        magnitude = math.sqrt(sum(val * val for val in vector))
        if magnitude == 0:
            return vector
        return [val / magnitude for val in vector]
    
    def resize_embedding(self, embedding: List[float]) -> List[float]:
        """Resize embedding to target dimension."""
        if len(embedding) == self.target_dimension:
            return embedding
        
        result = [0.0] * self.target_dimension
        ratio = len(embedding) / self.target_dimension
        
        for i in range(self.target_dimension):
            start = int(i * ratio)
            end = int((i + 1) * ratio)
            if end > len(embedding):
                end = len(embedding)
            
            if start < end:
                sum_val = sum(embedding[j] for j in range(start, end))
                result[i] = sum_val / (end - start)
        
        return self.normalize_vector(result)
    
    def embed(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            logger.info(f"Sending embedding request to endpoint: {self.embedding_endpoint}")
            logger.info(f"Using model: {self.embedding_model}")
            logger.info(f"Text length: {len(text)} characters")
            
            # Prepare request
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}'
            }
            
            data = {
                'model': self.embedding_model,
                'input': text
            }
            
            # Make request
            # Check if the endpoint already ends with /embeddings
            endpoint = self.embedding_endpoint
            if endpoint.endswith('/embeddings'):
                request_url = endpoint
            else:
                request_url = f"{endpoint}/embeddings"
                
            response = requests.post(
                request_url,
                headers=headers,
                json=data,
                timeout=30
            )
            
            if not response.ok:
                logger.warning(f"HTTP error! Status: {response.status_code}")
                logger.warning(f"Error response: {response.text}")
                return self.generate_random_embedding()
            
            response_data = response.json()
            
            # Check if we got a valid embedding in the expected OpenAI format
            if (not response_data or 
                not response_data.get('data') or 
                not response_data['data'][0] or 
                not response_data['data'][0].get('embedding')):
                logger.warning("Warning: Embedding API didn't return a valid embedding")
                logger.warning(f"Response: {response_data}")
                return self.generate_random_embedding()
            
            # Get the embedding array from the OpenAI-compatible format
            embedding = response_data['data'][0]['embedding']
            
            # Ensure we have the target dimensional vector
            resized_embedding = self.resize_embedding(embedding)
            
            logger.info(f"Successfully processed embedding with {len(resized_embedding)} dimensions")
            return resized_embedding
            
        except Exception as e:
            logger.error(f"Error fetching embedding from endpoint: {e}")
            return self.generate_random_embedding()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a given text."""
        return self.embed(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        return embeddings
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents with embeddings to the vector store."""
        try:
            # Generate embeddings for documents
            texts = [doc["content"] for doc in documents]
            embeddings = self.generate_embeddings_batch(texts)
            
            # Prepare documents with embeddings
            embedded_docs = []
            for i, doc in enumerate(documents):
                embedded_doc = {
                    "id": doc.get("id"),
                    "content": doc["content"],
                    "vector": embeddings[i],
                    "metadata": doc.get("metadata", {}),
                    "timestamp": doc.get("timestamp")
                }
                embedded_docs.append(embedded_doc)
            
            # Add to vector store
            return self.vector_store.add_documents(embedded_docs)
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def retrieve_similar_documents(
        self, 
        query: str, 
        k: int = None, 
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve similar documents for a query."""
        try:
            # Generate query embedding
            query_embedding = self.embed(query)
            
            # Search for similar documents
            results = self.vector_store.similarity_search(
                query_vector=query_embedding,
                k=k,
                filter_dict=filter_dict
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar documents: {e}")
            return []
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            List of documents with content and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embed(query)
            
            # Search using the vector store
            results = self.vector_store.similarity_search(
                query_vector=query_embedding,
                k=top_k
            )
            
            # Truncate content to reduce token usage
            for result in results:
                if len(result['content']) > 500:
                    result['content'] = result['content'][:500]
            
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []

    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Add a single document to the vector store.
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding for the document
            embedding = self.embed(content)
            
            # Prepare document
            doc = {
                "content": content,
                "vector": embedding,
                "metadata": metadata or {}
            }
            
            # Add to vector store
            return self.vector_store.add_documents([doc])
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    def retrieve_context(self, query: str, k: int = None) -> str:
        """Retrieve and format context for a query."""
        try:
            # Get similar documents
            similar_docs = self.retrieve_similar_documents(query, k)
            
            if not similar_docs:
                return "No relevant context found."
            
            # Format context
            context_parts = []
            for i, doc in enumerate(similar_docs, 1):
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "Unknown")
                
                context_part = f"[Context {i} - Source: {source}]\n{doc['content']}\n"
                context_parts.append(context_part)
            
            context = "\n".join(context_parts)
            logger.info(f"Retrieved context with {len(similar_docs)} documents, total length: {len(context)} characters")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return "Error retrieving context."
    
    def initialize_index(self, dimension: int = 384) -> bool:
        """Initialize the vector store index."""
        return self.vector_store.create_index(dimension)
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store."""
        return self.vector_store.get_document_count()
    
    def close(self) -> None:
        """Close the vector store connection."""
        if self.vector_store:
            self.vector_store.close()
