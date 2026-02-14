# Multi-Agent System Guide

## Overview

This guide explains how to use the multi-agent RAG system that has been implemented to replace the single-agent architecture.

## Architecture Changes

### Before (Single Agent)
- Single `Agent` class handling all responsibilities
- Direct embedding and retrieval in main flow
- Simple workflow execution

### After (Multi-Agent)
- **SupervisorAgent**: Orchestrates the entire workflow
- **KnowledgeAgent**: Manages knowledge embedding and change detection
- **RAGAgent**: Handles context retrieval and semantic search
- **MCPAgent**: Manages tool interactions and LLM communication

## Usage

### 1. Running the Multi-Agent System
```bash
pnpm dev
```

### 2. Embedding Knowledge (Standalone)
```bash
# Embed all knowledge files (markdown, text, JSON, and CSV) with change detection
pnpm embed-knowledge
```

### 3. Testing the System
```bash
pnpm test-agents
```

## Key Features

### Automatic Change Detection
- The KnowledgeAgent monitors file changes using hashes
- Only processes modified files for efficiency
- Maintains metadata across runs

### Intelligent Workflow
1. **Knowledge Check**: Automatically detects and embeds new/changed files
2. **Context Retrieval**: Uses RAG to find relevant information
3. **Task Execution**: Leverages MCP tools to complete tasks

### Error Handling
- Each agent handles its own errors gracefully
- Supervisor provides comprehensive error reporting
- Resource cleanup is handled properly

## Agent Responsibilities

### SupervisorAgent
- Initializes and coordinates all sub-agents
- Manages the complete workflow execution
- Provides task tracking and result summaries
- Handles cleanup and resource management

### KnowledgeAgent
- Scans knowledge directory for changes
- Embeds new or modified documents
- Supports multiple file formats (MD, TXT, JSON, CSV) in a unified process
- Maintains change detection metadata

### RAGAgent
- Performs semantic search using embeddings
- Retrieves relevant context for queries
- Supports advanced features like reranking
- Optimizes context for LLM consumption

### MCPAgent
- Manages LLM interactions with tool support
- Handles multi-turn conversations
- Processes tool calls through MCP protocol
- Maintains conversation context

## Configuration

The system uses the same environment variables as before:
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_BASE_URL`: Your model hosting endpoint
- `OPENSEARCH_ENDPOINT`: Your OpenSearch endpoint
- `AWS_REGION`: Your AWS region

## Benefits

1. **Modularity**: Each agent has a specific purpose
2. **Maintainability**: Easier to modify individual components
3. **Scalability**: Agents can be scaled independently
4. **Reliability**: Better error isolation and handling
5. **Extensibility**: Easy to add new agent types
