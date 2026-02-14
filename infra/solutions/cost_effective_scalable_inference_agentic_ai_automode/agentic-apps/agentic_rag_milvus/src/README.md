# RAG System with Custom Embedding Model

This project implements a Retrieval-Augmented Generation (RAG) system using a custom embedding model and Milvus vector database.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file with the following variables:
```
MILVUS_ADDRESS=your_milvus_address
MILVUS_USERNAME=your_milvus_username
MILVUS_PASSWORD=your_milvus_password
AWS_REGION=your_aws_region
```

## Embedding the CSV Data

To embed the `q_c_data.csv` file from the knowledge folder:

```bash
npm run embed-csv
```

This will process the CSV file, extract question-context pairs, and embed them using the custom embedding endpoint.

## Running the Application

To run the main application:

```bash
npm start
```

## Files

- `index.ts`: Main application entry point
- `EmbeddingRetriever.ts`: Handles embedding generation using the custom endpoint
- `MilvusVectorStore.ts`: Manages vector storage and retrieval in Milvus
- `Agent.ts`: Implements the agent that uses the RAG system
- `updateRAG.ts`: Script to process and embed the CSV data

## Custom Embedding Endpoint

The system uses a custom embedding endpoint at http://18.232.167.163:8080/v1/embeddings instead of AWS Bedrock.

Example request:
```bash
curl --request POST \
    --url http://18.232.167.163:8080/v1/embeddings \
    --header "Content-Type: application/json" \
    --data '{"content": "Your text here"}'
```
