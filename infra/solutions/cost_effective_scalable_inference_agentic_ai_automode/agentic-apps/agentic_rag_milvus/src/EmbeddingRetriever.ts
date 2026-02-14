import { logTitle } from "./utils";
import MilvusVectorStore from "./MilvusVectorStore";
import 'dotenv/config';
import fetch from 'node-fetch';

export default class EmbeddingRetriever {
    private embeddingModel: string;
    private vectorStore: MilvusVectorStore;
    private embeddingEndpoint: string;

    constructor(embeddingModel: string) {
        this.embeddingModel = embeddingModel;
        this.vectorStore = new MilvusVectorStore();
        this.embeddingEndpoint = 'http://18.232.167.163:8080/v1/embeddings';
    }

    async embedDocument(document: string) {
        logTitle('EMBEDDING DOCUMENT');
        const embedding = await this.embed(document);
        this.vectorStore.addEmbedding(embedding, document);
        return embedding;
    }

    async embedQuery(query: string) {
        logTitle('EMBEDDING QUERY');
        const embedding = await this.embed(query);
        return embedding;
    }

    private async embed(document: string): Promise<number[]> {
        try {
            console.log(`Sending embedding request to custom endpoint: ${this.embeddingEndpoint}`);
            console.log(`Document length: ${document.length} characters`);
            
            const response = await fetch(this.embeddingEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    content: document
                }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const responseBody = await response.json();
            
            // Check if we got a valid embedding
            // The response format is an array with objects containing the embedding
            if (Array.isArray(responseBody) && responseBody.length > 0 && responseBody[0].embedding) {
                // Extract the embedding from the first item in the array
                const embedding = responseBody[0].embedding;
                
                // Check if the embedding is a nested array and flatten it if needed
                const flatEmbedding = Array.isArray(embedding[0]) ? embedding[0] : embedding;
                
                console.log(`Successfully received embedding with ${flatEmbedding.length} dimensions`);
                return flatEmbedding;
            } else {
                console.log("Warning: Embedding API didn't return a valid embedding");
                console.log("Response:", JSON.stringify(responseBody, null, 2));
                // Return a small random embedding vector for testing purposes
                return Array(1536).fill(0).map(() => Math.random());
            }
        } catch (error) {
            console.error("Error fetching embedding from custom endpoint:", error);
            // Return a mock embedding in case of error
            return Array(1536).fill(0).map(() => Math.random());
        }
    }

    async retrieve(query: string, topK: number = 3): Promise<string[]> {
        console.log(`Starting retrieval for query: "${query.substring(0, 50)}..."`);
        
        const queryEmbedding = await this.embedQuery(query);
        console.log(`Generated query embedding with ${queryEmbedding.length} dimensions`);
        
        // Log a few values from the embedding to check consistency
        console.log(`Embedding sample values: [${queryEmbedding.slice(0, 5).join(', ')}]`);
        
        const results = await this.vectorStore.search(queryEmbedding, topK);
        console.log(`Search returned ${results.length} results`);
        
        if (results.length === 0) {
            console.log(`WARNING: No results found for query: "${query.substring(0, 50)}..."`);
        } else {
            console.log(`First result preview: "${results[0].substring(0, 100)}..."`);
        }
        
        return results;
    }
}
