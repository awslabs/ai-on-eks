# Agentic RAG with MCP and Custom Embedding

This project implements an augmented Large Language Model (LLM) system that combines Model Context Protocol (MCP) for tool usage and Retrieval Augmented Generation (RAG) for enhanced context awareness, all without relying on frameworks like LangChain or LlamaIndex.

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
   - The system loads knowledge documents and creates embeddings using a custom embedding endpoint
   - Embeddings are stored in a Milvus vector database
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

## Key Components

- **Agent**: Coordinates the overall workflow and manages tool usage
- **ChatOpenAI**: Handles interactions with the language model and tool calling
- **MCPClient**: Connects to MCP servers and manages tool calls
- **EmbeddingRetriever**: Creates and searches vector embeddings for relevant context
- **MilvusVectorStore**: Interfaces with Milvus for storing and retrieving embeddings

## Getting Started

### Prerequisites

- Node.js 18+
- pnpm or npm
- OpenAI API key
- Milvus database instance

### Installation

```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pnpm install

# Set up environment variables
# Create a .env file with:
# - OPENAI_API_KEY
# - OPENAI_BASE_URL (optional)
# - MILVUS_ADDRESS
```

### Usage

```bash
# Embed knowledge documents
pnpm embed-knowledge

# Embed CSV data (optional)
pnpm embed-csv

# Run the application
pnpm dev
```

## Example Use Case

The current implementation demonstrates a task where the agent:
1. Retrieves information about a user named "Antonette" from the knowledge base
2. Summarizes the information and creates a story about her
3. Saves the output to a markdown file using the filesystem MCP tool

## Extending the System

This modular architecture can be easily extended:
- Add more MCP servers for additional tool capabilities
- Implement advanced Milvus features like filtering and hybrid search
- Add more sophisticated RAG techniques like reranking or chunking
- Implement conversation history for multi-turn interactions
- Deploy as a service with API endpoints
- Integrate with different LLM providers
- Scale the vector database for production workloads
