#!/usr/bin/env python3
"""
Tavily Web Search MCP Server

This MCP server provides web search capabilities using the Tavily API.
It's designed to be called when RAG retrieval relevance is low.
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from fastmcp import FastMCP
import httpx
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Loaded environment variables from {env_file}")
else:
    print(f"⚠️  No .env file found at {env_file}")

class TavilySearchResult(BaseModel):
    """Model for Tavily search results"""
    title: str
    url: str
    content: str
    score: float
    published_date: Optional[str] = None

class TavilySearchResponse(BaseModel):
    """Model for complete Tavily response"""
    query: str
    results: List[TavilySearchResult]
    answer: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None

# Initialize FastMCP server
mcp = FastMCP("Tavily Web Search Server")

class TavilyClient:
    """Client for interacting with Tavily API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com"
        
    async def search(
        self, 
        query: str, 
        search_depth: str = "basic",
        max_results: int = 5,
        include_answer: bool = True,
        include_raw_content: bool = False,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None
    ) -> TavilySearchResponse:
        """
        Perform web search using Tavily API
        
        Args:
            query: Search query
            search_depth: "basic" or "advanced"
            max_results: Maximum number of results (1-20)
            include_answer: Whether to include AI-generated answer
            include_raw_content: Whether to include raw HTML content
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            
        Returns:
            TavilySearchResponse with search results
        """
        
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": min(max_results, 20),
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }
        
        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                data = response.json()
                
                # Parse results
                results = []
                for result in data.get("results", []):
                    results.append(TavilySearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("content", ""),
                        score=result.get("score", 0.0),
                        published_date=result.get("published_date")
                    ))
                
                return TavilySearchResponse(
                    query=query,
                    results=results,
                    answer=data.get("answer"),
                    follow_up_questions=data.get("follow_up_questions", [])
                )
                
            except httpx.HTTPStatusError as e:
                raise Exception(f"Tavily API error: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"Search failed: {str(e)}")

# Initialize Tavily client
tavily_client = None

def get_tavily_client() -> TavilyClient:
    """Get or create Tavily client"""
    global tavily_client
    if tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        tavily_client = TavilyClient(api_key)
    return tavily_client

@mcp.tool(description="Search the web for real-time information using Tavily API")
async def web_search(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True
) -> str:
    """
    Search the web for real-time information when RAG retrieval relevance is low.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)
        search_depth: Search depth - "basic" or "advanced"
        include_answer: Whether to include AI-generated answer summary
        
    Returns:
        JSON string containing search results and answer
    """
    try:
        client = get_tavily_client()
        
        # Perform search
        response = await client.search(
            query=query,
            max_results=min(max_results, 10),
            search_depth=search_depth,
            include_answer=include_answer
        )
        
        # Format response for the agent
        formatted_response = {
            "query": response.query,
            "answer": response.answer,
            "results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    "score": result.score,
                    "published_date": result.published_date
                }
                for result in response.results
            ],
            "follow_up_questions": response.follow_up_questions or []
        }
        
        return json.dumps(formatted_response, indent=2)
        
    except Exception as e:
        error_response = {
            "error": f"Web search failed: {str(e)}",
            "query": query,
            "results": [],
            "answer": None
        }
        return json.dumps(error_response, indent=2)

@mcp.tool(description="Search for recent news and current events")
async def news_search(
    query: str,
    max_results: int = 5,
    days_back: int = 7
) -> str:
    """
    Search for recent news and current events.
    
    Args:
        query: The news search query
        max_results: Maximum number of results to return
        days_back: How many days back to search for news
        
    Returns:
        JSON string containing recent news results
    """
    try:
        client = get_tavily_client()
        
        # Add time constraint to query for recent news
        time_constrained_query = f"{query} recent news last {days_back} days"
        
        response = await client.search(
            query=time_constrained_query,
            max_results=min(max_results, 10),
            search_depth="basic",
            include_answer=True
        )
        
        # Filter for more recent results and news sources
        news_domains = ["reuters.com", "bbc.com", "cnn.com", "ap.org", "npr.org", "bloomberg.com"]
        
        formatted_response = {
            "query": response.query,
            "answer": response.answer,
            "news_results": [
                {
                    "title": result.title,
                    "url": result.url,
                    "content": result.content[:400] + "..." if len(result.content) > 400 else result.content,
                    "score": result.score,
                    "published_date": result.published_date,
                    "is_news_source": any(domain in result.url for domain in news_domains)
                }
                for result in response.results
            ],
            "follow_up_questions": response.follow_up_questions or []
        }
        
        return json.dumps(formatted_response, indent=2)
        
    except Exception as e:
        error_response = {
            "error": f"News search failed: {str(e)}",
            "query": query,
            "news_results": [],
            "answer": None
        }
        return json.dumps(error_response, indent=2)

@mcp.tool(description="Get health check status of the Tavily search service")
async def health_check() -> str:
    """
    Check if the Tavily API service is available and working.
    
    Returns:
        JSON string with health status
    """
    try:
        client = get_tavily_client()
        
        # Perform a simple test search
        response = await client.search(
            query="test",
            max_results=1,
            include_answer=False
        )
        
        status = {
            "status": "healthy",
            "service": "Tavily Web Search",
            "api_accessible": True,
            "test_query_successful": len(response.results) > 0,
            "timestamp": time.time()
        }
        
        return json.dumps(status, indent=2)
        
    except Exception as e:
        status = {
            "status": "unhealthy",
            "service": "Tavily Web Search",
            "api_accessible": False,
            "error": str(e),
            "timestamp": time.time()
        }
        return json.dumps(status, indent=2)

if __name__ == "__main__":
    # Run the MCP server
    print("Starting Tavily Web Search MCP Server...")
    print("Available tools:")
    print("- web_search: General web search with AI-generated answers")
    print("- news_search: Recent news and current events search")
    print("- health_check: Service health status")
    print("\nMCP server will be available at http://localhost:8001/mcp")
    
    mcp.run(transport="streamable-http", port=8001)
