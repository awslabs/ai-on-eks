# Augmented LLM with MCP and RAG

This project demonstrates a framework-independent implementation of an augmented Large Language Model (LLM) system that combines Model Context Protocol (MCP) for tool usage and Retrieval Augmented Generation (RAG) for enhanced context awareness.

## Project Overview

This application creates an AI agent that can:
1. Retrieve relevant information from a knowledge base using vector embeddings
2. Interact with external tools through the Model Context Protocol (MCP)
3. Generate responses based on both the retrieved context and tool interactions
4. Complete tasks like summarizing content and saving results to files

## Architecture

The system is built with a modular architecture consisting of these key components:

```
Agent → Manages the overall workflow and coordinates components
  ├── ChatOpenAI → Handles LLM interactions and tool calling
  ├── MCPClient(s) → Connects to MCP servers for tool access
  └── EmbeddingRetriever → Performs vector search for relevant context
      └── VectorStore → Stores and searches document embeddings
```

## Workflow Explanation

1. **Initialization**:
   - The system loads knowledge documents and creates embeddings using AWS Bedrock
   - Embeddings are stored in an in-memory vector store
   - MCP clients are initialized to connect to tool servers

2. **RAG Process**:
   - When a query is received, it's converted to an embedding
   - The system searches for the most relevant documents using cosine similarity
   - Retrieved documents are combined to form context for the LLM

3. **Agent Execution**:
   - The agent initializes with the LLM, MCP clients, and retrieved context
   - The user query is sent to the LLM along with the context
   - The LLM generates responses and may request tool calls

4. **Tool Usage**:
   - When the LLM requests a tool, the agent routes the call to the appropriate MCP client
   - The MCP client executes the tool and returns results
   - Results are fed back to the LLM to continue the conversation

5. **Output Generation**:
   - The LLM generates a final response incorporating tool results and context
   - In the example task, it creates a markdown file with information about "Antonette"

## Key Technologies

- **LLM Integration**: Uses OpenAI API for language model capabilities
- **MCP Implementation**: Connects to MCP servers for filesystem operations
- **Vector Embeddings**: Uses AWS Bedrock for generating embeddings
- **Vector Search**: Implements cosine similarity for finding relevant documents

## Implementation Details

- **Framework Independence**: Built without relying on frameworks like LangChain or LlamaIndex
- **Modular Design**: Components are separated for easy maintenance and extension
- **AWS Integration**: Uses AWS Bedrock for embedding generation
- **Tool Orchestration**: Manages tool calls and responses through MCP protocol

## Example Use Case

The current implementation demonstrates a task where the agent:
1. Retrieves information about a user named "Antonette" from the knowledge base
2. Summarizes the information and creates a story about her
3. Saves the output to a markdown file using the filesystem MCP tool

This architecture can be extended to support various tasks requiring context-aware responses and tool usage.
