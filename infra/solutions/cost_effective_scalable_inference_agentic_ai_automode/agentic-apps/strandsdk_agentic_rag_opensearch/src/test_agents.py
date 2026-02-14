#!/usr/bin/env python3
"""
Agent Testing Script

This script tests the multi-agent system with various queries to ensure
all components are working correctly, including Langfuse integration.
"""

import sys
import logging
from typing import List, Dict, Any
from .config import config
from .utils.logging import setup_logging, log_title
from .agents.supervisor_agent import supervisor_agent, supervisor_agent_with_langfuse
from .agents.knowledge_agent import knowledge_agent, knowledge_agent_with_langfuse
from .agents.mcp_agent import mcp_agent, mcp_agent_with_langfuse

def main():
    """Main function for testing agents."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Validate configuration
        config.validate_config()
        
        log_title("AGENT TESTING SCRIPT")
        logger.info("Starting agent testing process")
        
        # Test individual agents first
        test_individual_agents()
        
        # Test supervisor agent
        test_supervisor_agent()
        
        print("\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Agent testing failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)

def test_individual_agents():
    """Test individual agents."""
    log_title("INDIVIDUAL AGENT TESTS")
    
    # Test Knowledge Agent
    print("🧠 Testing Knowledge Agent...")
    try:
        if config.is_langfuse_enabled():
            knowledge_response = knowledge_agent_with_langfuse("Please scan the knowledge directory and report what files are available.")
            print(f"✅ Knowledge Agent Response (with Langfuse): {str(knowledge_response)[:200]}...")
        else:
            knowledge_response = knowledge_agent("Please scan the knowledge directory and report what files are available.")
            print(f"✅ Knowledge Agent Response: {str(knowledge_response)[:200]}...")
    except Exception as e:
        print(f"❌ Knowledge Agent failed: {e}")
    
    # Test MCP Agent
    print("\n🔧 Testing MCP Agent...")
    try:
        if config.is_langfuse_enabled():
            mcp_response = mcp_agent_with_langfuse("Please help me understand what tools are available for file operations.")
            print(f"✅ MCP Agent Response (with Langfuse): {str(mcp_response)[:200]}...")
        else:
            mcp_response = mcp_agent("Please help me understand what tools are available for file operations.")
            print(f"✅ MCP Agent Response: {str(mcp_response)[:200]}...")
    except Exception as e:
        print(f"❌ MCP Agent failed: {e}")

def test_supervisor_agent():
    """Test the supervisor agent with various queries."""
    log_title("SUPERVISOR AGENT TESTS")
    
    test_queries = [
        "What is the status of the knowledge base?",
        "Can you help me understand what files are in the knowledge directory?",
        "Please search for information about Bell's palsy if available."
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {query}")
        
        try:
            if config.is_langfuse_enabled():
                response = supervisor_agent_with_langfuse(query)
                print(f"✅ Success (with Langfuse): {str(response)[:300]}...")
            else:
                response = supervisor_agent(query)
                print(f"✅ Success: {str(response)[:300]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
