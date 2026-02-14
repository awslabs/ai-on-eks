import EmbeddingRetriever from "../EmbeddingRetriever";
import { logTitle } from "../utils";

export default class RAGAgent {
    private embeddingRetriever: EmbeddingRetriever | null = null;
    private model: string;

    constructor(model: string = 'Qwen/QwQ-32B-AWQ') {
        this.model = model;
    }

    async init() {
        logTitle('INITIALIZING RAG AGENT');
        
        // Initialize the embedding retriever with llamacpp-embedding model
        this.embeddingRetriever = new EmbeddingRetriever("llamacpp-embedding");
        
        console.log('RAG Agent initialized');
    }

    async close() {
        // Close embedding retriever if needed
        if (this.embeddingRetriever) {
            // @ts-ignore - Access private property for cleanup
            if (this.embeddingRetriever.vectorStore && typeof this.embeddingRetriever.vectorStore.close === 'function') {
                // @ts-ignore
                await this.embeddingRetriever.vectorStore.close();
            }
        }
    }

    async retrieveContext(query: string, topK: number = 5): Promise<string> {
        logTitle('RAG CONTEXT RETRIEVAL');
        console.log(`Query: ${query}`);
        console.log(`Retrieving top ${topK} relevant documents`);
        
        if (!this.embeddingRetriever) {
            throw new Error('RAG Agent not initialized');
        }

        try {
            // Retrieve relevant documents
            const documents = await this.embeddingRetriever.retrieve(query, topK);
            
            // Combine documents into context
            const context = documents.join('\n\n---\n\n');
            
            console.log(`Retrieved ${documents.length} relevant documents`);
            console.log('Context preview:', context.substring(0, 200) + '...');
            
            return context;
        } catch (error) {
            console.error('Error in context retrieval:', error);
            throw error;
        }
    }

    async retrieveContextWithMetadata(query: string, topK: number = 5): Promise<{
        context: string;
        documents: string[];
        metadata: any[];
    }> {
        logTitle('RAG CONTEXT RETRIEVAL WITH METADATA');
        console.log(`Query: ${query}`);
        
        if (!this.embeddingRetriever) {
            throw new Error('RAG Agent not initialized');
        }

        try {
            // Retrieve relevant documents
            const documents = await this.embeddingRetriever.retrieve(query, topK);
            
            // For now, we don't have detailed metadata, but this structure allows for future enhancement
            const metadata = documents.map((doc, index) => ({
                index,
                length: doc.length,
                preview: doc.substring(0, 100)
            }));
            
            const context = documents.join('\n\n---\n\n');
            
            console.log(`Retrieved ${documents.length} relevant documents with metadata`);
            
            return {
                context,
                documents,
                metadata
            };
        } catch (error) {
            console.error('Error in context retrieval with metadata:', error);
            throw error;
        }
    }

    async semanticSearch(query: string, filters?: any): Promise<string[]> {
        logTitle('SEMANTIC SEARCH');
        console.log(`Semantic search query: ${query}`);
        
        if (!this.embeddingRetriever) {
            throw new Error('RAG Agent not initialized');
        }

        try {
            // For now, use the standard retrieve method
            // In the future, this could be enhanced with filtering capabilities
            const documents = await this.embeddingRetriever.retrieve(query, 10);
            
            console.log(`Found ${documents.length} semantically similar documents`);
            
            return documents;
        } catch (error) {
            console.error('Error in semantic search:', error);
            throw error;
        }
    }

    async hybridSearch(query: string, keywordWeight: number = 0.3, semanticWeight: number = 0.7): Promise<string[]> {
        logTitle('HYBRID SEARCH');
        console.log(`Hybrid search query: ${query}`);
        console.log(`Keyword weight: ${keywordWeight}, Semantic weight: ${semanticWeight}`);
        
        if (!this.embeddingRetriever) {
            throw new Error('RAG Agent not initialized');
        }

        try {
            // For now, this is just semantic search
            // In a full implementation, this would combine keyword and semantic search
            const documents = await this.embeddingRetriever.retrieve(query, 10);
            
            console.log(`Hybrid search returned ${documents.length} documents`);
            
            return documents;
        } catch (error) {
            console.error('Error in hybrid search:', error);
            throw error;
        }
    }

    async rerank(query: string, documents: string[], topK: number = 5): Promise<string[]> {
        logTitle('DOCUMENT RERANKING');
        console.log(`Reranking ${documents.length} documents for query: ${query}`);
        
        // Simple reranking based on query term frequency
        // In a production system, this would use a more sophisticated reranking model
        const scoredDocs = documents.map(doc => {
            const queryTerms = query.toLowerCase().split(/\s+/);
            const docLower = doc.toLowerCase();
            
            let score = 0;
            for (const term of queryTerms) {
                const matches = (docLower.match(new RegExp(term, 'g')) || []).length;
                score += matches;
            }
            
            return { doc, score };
        });
        
        // Sort by score and return top K
        const reranked = scoredDocs
            .sort((a, b) => b.score - a.score)
            .slice(0, topK)
            .map(item => item.doc);
        
        console.log(`Reranked to top ${reranked.length} documents`);
        
        return reranked;
    }

    getStats(): {
        model: string;
        initialized: boolean;
    } {
        return {
            model: this.model,
            initialized: this.embeddingRetriever !== null
        };
    }
}
