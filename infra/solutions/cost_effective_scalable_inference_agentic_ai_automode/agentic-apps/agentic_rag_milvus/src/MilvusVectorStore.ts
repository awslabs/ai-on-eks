import { MilvusClient, DataType, InsertReq, SearchParam } from '@zilliz/milvus2-sdk-node';
import { VectorStoreItem } from './VectorStore';
import 'dotenv/config';

export default class MilvusVectorStore {
    private client: MilvusClient;
    private collectionName: string = 'rag_documents_384d';
    private dimension: number = 384; // Changed to match the custom embedding endpoint dimensions

    constructor() {
        // Connect to Milvus service through NLB
        this.client = new MilvusClient({
            address: process.env.MILVUS_ADDRESS || '',
            username: process.env.MILVUS_USERNAME || '',
            password: process.env.MILVUS_PASSWORD || '',
        });
        this.initCollection();
    }

    private async initCollection() {
        try {
            // Check if collection exists
            const hasCollection = await this.client.hasCollection({
                collection_name: this.collectionName,
            });

            if (!hasCollection.value) {
                // Create collection if it doesn't exist
                await this.client.createCollection({
                    collection_name: this.collectionName,
                    fields: [
                        {
                            name: 'id',
                            data_type: DataType.Int64,
                            is_primary_key: true,
                            autoID: true,
                        },
                        {
                            name: 'embedding',
                            data_type: DataType.FloatVector,
                            dim: this.dimension,
                        },
                        {
                            name: 'document',
                            data_type: DataType.VarChar,
                            max_length: 65535,
                        },
                    ],
                });

                // Create index for vector search
                await this.client.createIndex({
                    collection_name: this.collectionName,
                    field_name: 'embedding',
                    index_type: 'HNSW',
                    metric_type: 'COSINE',
                    params: { M: 8, efConstruction: 64 },
                });

                // Load collection into memory
                await this.client.loadCollection({
                    collection_name: this.collectionName,
                });
            }
        } catch (error) {
            console.error('Error initializing Milvus collection:', error);
        }
    }

    async addEmbedding(embedding: number[], document: string) {
        try {
            // Create a new client for each insertion to avoid pool draining issues
            const insertClient = new MilvusClient({
                address: process.env.MILVUS_ADDRESS || '',
                username: process.env.MILVUS_USERNAME || '',
                password: process.env.MILVUS_PASSWORD || '',
            });
            
            const insertData: InsertReq = {
                collection_name: this.collectionName,
                fields_data: [{
                    embedding: embedding,
                    document: document,
                }],
            };
            
            await insertClient.insert(insertData);
            
            // Close the client after insertion
            await insertClient.closeConnection();
        } catch (error) {
            console.error('Error adding embedding to Milvus:', error);
        }
    }

    async search(queryEmbedding: number[], topK: number = 3): Promise<string[]> {
        try {
            console.log(`Searching Milvus with topK=${topK}`);
            
            // Verify the embedding dimensions
            console.log(`Query embedding dimensions: ${queryEmbedding.length}`);
            if (queryEmbedding.length !== this.dimension) {
                console.error(`Dimension mismatch: Query embedding has ${queryEmbedding.length} dimensions, but collection expects ${this.dimension}`);
            }
            
            const searchParams: SearchParam = {
                collection_name: this.collectionName,
                vector: queryEmbedding,
                output_fields: ['document'],
                limit: topK,
                params: { ef: 64 },
            };
            
            console.log(`Executing search with params: ${JSON.stringify({
                collection_name: searchParams.collection_name,
                limit: searchParams.limit,
                params: searchParams.params
            })}`);
            
            const searchResult = await this.client.search(searchParams);
            
            console.log(`Search completed. Results: ${searchResult?.results?.length || 0}`);
            
            if (searchResult && searchResult.results && searchResult.results.length > 0) {
                console.log(`Found ${searchResult.results.length} matching documents`);
                return searchResult.results.map(item => item.document as string);
            } else {
                console.log(`No matching documents found. Search result: ${JSON.stringify(searchResult)}`);
                return [];
            }
        } catch (error) {
            console.error('Error searching in Milvus:', error);
            console.error('Error details:', JSON.stringify(error, null, 2));
            return [];
        }
    }

    async close() {
        try {
            await this.client.closeConnection();
        } catch (error) {
            console.error('Error closing Milvus connection:', error);
        }
    }
}
