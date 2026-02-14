export interface VectorStoreItem {
    embedding: number[];
    document: string;
}

export interface VectorStore {
    addEmbedding(embedding: number[], document: string): Promise<void>;
    search(queryEmbedding: number[], topK: number): Promise<string[]>;
    close?(): Promise<void>;
}
