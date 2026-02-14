#!/usr/bin/env python3
"""
Knowledge Embedding Script

This script processes all documents in the knowledge directory and creates
embeddings for them in the OpenSearch vector store.
"""

import sys
import logging
from pathlib import Path
from ..config import config
from ..utils.logging import setup_logging, log_title
from ..agents.knowledge_agent import knowledge_agent

def main():
    """Main function for embedding knowledge."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        config.validate_config()
        
        log_title("KNOWLEDGE EMBEDDING SCRIPT")
        logger.info("Starting knowledge embedding process")
        logger.info(f"Knowledge Directory: {config.KNOWLEDGE_DIR}")
        logger.info(f"OpenSearch Endpoint: {config.OPENSEARCH_ENDPOINT}")
        logger.info(f"Vector Index: {config.VECTOR_INDEX_NAME}")
        
        # Check if knowledge directory exists
        knowledge_path = Path(config.KNOWLEDGE_DIR)
        if not knowledge_path.exists():
            logger.error(f"Knowledge directory does not exist: {config.KNOWLEDGE_DIR}")
            sys.exit(1)
        
        # Get knowledge statistics before embedding
        print("\n📊 Getting knowledge statistics...")
        stats_result = knowledge_agent("get_stats")
        if stats_result.get("success"):
            stats = stats_result.get("stats", {})
            print(f"Total files: {stats.get('total_files', 0)}")
            print(f"File types: {stats.get('file_types', {})}")
            print(f"Current vector store count: {stats.get('vector_store_count', 0)}")
        
        # Check for changes
        print("\n🔍 Checking for knowledge changes...")
        check_result = knowledge_agent("check_changes")
        
        if not check_result.get("success"):
            logger.error(f"Failed to check for changes: {check_result.get('message')}")
            sys.exit(1)
        
        has_changes = check_result.get("has_changes", False)
        print(f"Changes detected: {has_changes}")
        
        # Embed knowledge (force embedding regardless of changes)
        print("\n🚀 Starting knowledge embedding...")
        embed_result = knowledge_agent("embed_knowledge")
        
        if embed_result.get("success"):
            print("✅ Knowledge embedding completed successfully!")
            
            # Get updated statistics
            print("\n📊 Updated knowledge statistics...")
            updated_stats_result = knowledge_agent("get_stats")
            if updated_stats_result.get("success"):
                updated_stats = updated_stats_result.get("stats", {})
                print(f"Vector store count after embedding: {updated_stats.get('vector_store_count', 0)}")
        else:
            print(f"❌ Knowledge embedding failed: {embed_result.get('message')}")
            sys.exit(1)
        
        print("\n🎉 Knowledge embedding process completed!")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Knowledge embedding failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
