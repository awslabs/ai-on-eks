#!/usr/bin/env python3
"""
Test script for Web Search Integration with Tavily MCP Server
"""

import os
import sys
import json
import time
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.supervisor_agent import supervisor_agent
from src.utils.logging import setup_logging

def test_rag_relevance_scoring():
    """Test RAG relevance scoring and web search triggering"""
    print("🧪 Testing RAG Relevance Scoring and Web Search Integration")
    print("=" * 60)
    
    # Test cases with expected low relevance (should trigger web search)
    low_relevance_queries = [
        "What happened in the news today?",
        "Current stock market prices",
        "Latest AI developments in 2024",
        "Recent weather updates"
    ]
    
    # Test cases with expected high relevance (should use RAG)
    high_relevance_queries = [
        "What is Bell's palsy?",  # Assuming medical knowledge is in the knowledge base
        "Symptoms of facial paralysis",
        "Treatment for neurological conditions"
    ]
    
    print("\n📊 Testing Low Relevance Queries (Should Trigger Web Search):")
    print("-" * 50)
    
    for query in low_relevance_queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = supervisor_agent(query)
            print(f"✅ Response received (length: {len(result)} chars)")
            
            # Check if web search was mentioned in the response
            if "web" in result.lower() or "search" in result.lower():
                print("🌐 Web search likely used")
            else:
                print("📚 RAG likely used")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)  # Rate limiting
    
    print("\n📚 Testing High Relevance Queries (Should Use RAG):")
    print("-" * 50)
    
    for query in high_relevance_queries:
        print(f"\n🔍 Query: {query}")
        try:
            result = supervisor_agent(query)
            print(f"✅ Response received (length: {len(result)} chars)")
            
            # Check source indicators
            if "knowledge base" in result.lower() or "retrieved" in result.lower():
                print("📚 RAG likely used")
            elif "web" in result.lower():
                print("🌐 Web search used (unexpected for this query)")
            else:
                print("❓ Source unclear")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)  # Rate limiting

def test_direct_web_search():
    """Test direct web search functionality"""
    print("\n🌐 Testing Direct Web Search Tools")
    print("=" * 40)
    
    # Test web search tool directly
    test_queries = [
        "Latest developments in artificial intelligence",
        "Current events in technology",
        "Recent scientific discoveries"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing web search for: {query}")
        try:
            # This should trigger web search regardless of RAG relevance
            prompt = f"Use web search to find information about: {query}"
            result = supervisor_agent(prompt)
            print(f"✅ Web search result received (length: {len(result)} chars)")
            
            # Look for web search indicators
            if any(indicator in result.lower() for indicator in ["url", "http", "web", "search"]):
                print("🌐 Web search successfully used")
            else:
                print("❓ Web search usage unclear")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)

def test_news_search():
    """Test news search functionality"""
    print("\n📰 Testing News Search")
    print("=" * 30)
    
    news_queries = [
        "Recent technology news",
        "Latest AI breakthroughs",
        "Current events in science"
    ]
    
    for query in news_queries:
        print(f"\n📰 Testing news search for: {query}")
        try:
            prompt = f"Find recent news about: {query}"
            result = supervisor_agent(prompt)
            print(f"✅ News search result received (length: {len(result)} chars)")
            
            # Look for news indicators
            if any(indicator in result.lower() for indicator in ["news", "recent", "published", "article"]):
                print("📰 News search successfully used")
            else:
                print("❓ News search usage unclear")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)

def test_hybrid_responses():
    """Test hybrid responses combining RAG and web search"""
    print("\n🔄 Testing Hybrid RAG + Web Search Responses")
    print("=" * 45)
    
    hybrid_queries = [
        "What is machine learning and what are the latest developments?",
        "Explain neural networks and recent breakthroughs",
        "What are the current trends in AI research?"
    ]
    
    for query in hybrid_queries:
        print(f"\n🔄 Testing hybrid query: {query}")
        try:
            result = supervisor_agent(query)
            print(f"✅ Hybrid response received (length: {len(result)} chars)")
            
            # Check for both RAG and web search indicators
            has_rag = any(indicator in result.lower() for indicator in ["knowledge", "retrieved", "database"])
            has_web = any(indicator in result.lower() for indicator in ["web", "search", "recent", "current"])
            
            if has_rag and has_web:
                print("🎯 Hybrid response detected (RAG + Web)")
            elif has_rag:
                print("📚 RAG-only response")
            elif has_web:
                print("🌐 Web-only response")
            else:
                print("❓ Response source unclear")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(2)

def main():
    """Run all web search integration tests"""
    print("🚀 Starting Web Search Integration Tests")
    print("=" * 50)
    
    # Setup logging
    setup_logging()
    
    # Check if Tavily API key is configured
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not found in environment")
        print("Web search functionality may not work properly")
        print("Please set TAVILY_API_KEY in your .env file")
        return
    
    try:
        # Run tests
        test_rag_relevance_scoring()
        test_direct_web_search()
        test_news_search()
        test_hybrid_responses()
        
        print("\n🎉 All tests completed!")
        print("\n📋 Test Summary:")
        print("- RAG relevance scoring and automatic web search triggering")
        print("- Direct web search functionality")
        print("- News search capabilities")
        print("- Hybrid RAG + web search responses")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
