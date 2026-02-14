#!/usr/bin/env python3
"""
Multi-Agent RAG System with MCP and OpenSearch using Strands SDK

A sophisticated multi-agent system that combines:
- Knowledge management with change detection
- RAG (Retrieval Augmented Generation) with OpenSearch
- MCP (Model Context Protocol) tool integration
- Strands SDK agent orchestration
- Langfuse observability integration
"""

# Import global async cleanup FIRST to suppress warnings
from .utils.global_async_cleanup import setup_global_async_cleanup

import sys
import logging
from typing import Optional
from .config import config
from .utils.logging import setup_logging, log_title
from .agents.supervisor_agent import supervisor_agent
from .agents.knowledge_agent import knowledge_agent
from .agents.mcp_agent import mcp_agent

def main():
    """Main application entry point."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        config.validate_config()
        
        log_title("MULTI-AGENT RAG SYSTEM STARTUP")
        logger.info("Starting Multi-Agent RAG System with Strands SDK")
        logger.info(f"OpenSearch Endpoint: {config.OPENSEARCH_ENDPOINT}")
        logger.info(f"Knowledge Directory: {config.KNOWLEDGE_DIR}")
        logger.info(f"Reasoning Model: {config.REASONING_MODEL}")
        logger.info(f"Embedding Model: {config.EMBEDDING_MODEL}")
        logger.info(f"LiteLLM Endpoint: {config.LITELLM_BASE_URL}")
        logger.info(f"Langfuse Enabled: {config.is_langfuse_enabled()}")
        
        # Interactive mode
        run_interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\nExiting gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        sys.exit(1)

def run_interactive_mode():
    """Run the application in interactive mode."""
    logger = logging.getLogger(__name__)  # Add logger definition
    
    log_title("INTERACTIVE MODE")
    print("🤖 Multi-Agent RAG System Ready!")
    print("Ask questions and I'll use my specialized agents to help you.")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.\n")
    
    while True:
        try:
            # Get user input
            user_input = input("❓ Your question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                print("Please enter a question or type 'exit' to quit.")
                continue
            
            # Process the query using the supervisor agent
            print("\n🔄 Processing your request...")
            
            try:
                # Limit query length to avoid context window issues
                if len(user_input) > 500:
                    print("⚠️ Query is too long, truncating to 500 characters...")
                    user_input = user_input[:500]
                
                # Add debug logging
                logger.info(f"Starting agent processing for query: {user_input[:50]}...")
                
                # Create a fresh agent instance for each query to avoid context accumulation
                from .agents.supervisor_agent import create_fresh_supervisor_agent
                fresh_agent = create_fresh_supervisor_agent()
                
                # Use the fresh agent instance (no conversation history)
                response = fresh_agent(user_input)
                
                logger.info("Agent processing completed")
                
                # Ensure response is fully processed before displaying
                if response is None:
                    response = "No response received from agent."
                
                response_str = str(response).strip()
                if not response_str:
                    response_str = "Agent completed processing but returned empty response."
                
                print(f"\n🤖 Response:\n{response_str}")
                
                # Add a small delay to ensure all background processes complete
                import time
                time.sleep(0.5)
                
                logger.info("Response display completed, ready for next input")
                
            except KeyboardInterrupt:
                print("\n⚠️ Processing interrupted by user.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Error processing query: {e}")
            
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except EOFError:
            print("\n\nInput stream closed. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different question.\n")
            logger.error(f"Unexpected error in interactive mode: {e}")

def run_single_query(query: str) -> Optional[str]:
    """Run a single query and return the result."""
    try:
        config.validate_config()
        
        # Limit query length to avoid context window issues
        if len(query) > 500:
            logging.warning("Query too long, truncating to 500 characters")
            query = query[:500]
        
        # Create a fresh agent instance for this single query
        from .agents.supervisor_agent import create_fresh_supervisor_agent
        fresh_agent = create_fresh_supervisor_agent()
        
        # Use the fresh agent (no conversation history)
        response = fresh_agent(query)
            
        # Limit response length if needed
        response_str = str(response)
        if len(response_str) > 4000:
            logging.warning("Response too long, truncating to 4000 characters")
            response_str = response_str[:4000] + "... [Response truncated due to length]"
            
        return response_str
    except Exception as e:
        logging.error(f"Single query execution failed: {e}")
        return f"Error processing query: {str(e)}"

if __name__ == "__main__":
    main()
