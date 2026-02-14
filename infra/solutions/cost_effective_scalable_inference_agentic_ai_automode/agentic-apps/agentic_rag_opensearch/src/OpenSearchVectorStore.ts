import { Client } from '@opensearch-project/opensearch';
import { VectorStore } from './VectorStore';
import 'dotenv/config';
import AWS from 'aws-sdk';
import { AwsSigv4Signer } from '@opensearch-project/opensearch/aws';

export default class OpenSearchVectorStore implements VectorStore {
    private client: Client;
    private indexName: string = 'rag_documents';
    private dimension: number = 384; // Default dimension for embeddings

    constructor() {
        // Get configuration from environment variables
        const region = process.env.AWS_REGION;
        const opensearchEndpoint = process.env.OPENSEARCH_ENDPOINT;
        
        if (!region || !opensearchEndpoint) {
            throw new Error('AWS_REGION and OPENSEARCH_ENDPOINT environment variables must be set');
        }

        // Create AWS credentials
        const credentials = {
            accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
            secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || ''
        };

        // Initialize the OpenSearch client with AWS Signature V4 authentication
        this.client = new Client({
            ...AwsSigv4Signer({
                region: region,
                service: 'es',
                credentials: credentials
            }),
            node: opensearchEndpoint
        });
        
        this.initIndex();
    }

    private async initIndex() {
        try {
            // Check if index exists
            const indexExists = await this.client.indices.exists({
                index: this.indexName
            });
            
            if (!indexExists.body) {
                console.log(`Creating index ${this.indexName}`);
                
                // Create index with mapping for vector field
                await this.client.indices.create({
                    index: this.indexName,
                    body: {
                        settings: {
                            "index.knn": true,
                            "index.knn.space_type": "cosinesimil"
                        },
                        mappings: {
                            properties: {
                                embedding: {
                                    type: 'knn_vector',
                                    dimension: this.dimension,
                                    method: {
                                        name: 'hnsw',
                                        space_type: 'cosinesimil',
                                        engine: 'nmslib',
                                        parameters: {
                                            ef_construction: 128,
                                            m: 16
                                        }
                                    }
                                },
                                document: {
                                    type: 'text',
                                    store: true
                                }
                            }
                        }
                    }
                });
                
                console.log(`Index ${this.indexName} created successfully`);
            } else {
                console.log(`Index ${this.indexName} already exists`);
            }
        } catch (error) {
            console.error('Error initializing OpenSearch index:', error);
            throw error;
        }
    }

    async addEmbedding(embedding: number[], document: string): Promise<void> {
        try {
            await this.client.index({
                index: this.indexName,
                body: {
                    embedding: embedding,
                    document: document
                },
                refresh: true // Make the document immediately searchable
            });
            console.log('Document added to OpenSearch');
        } catch (error) {
            console.error('Error adding embedding to OpenSearch:', error);
            throw error;
        }
    }

    async search(queryEmbedding: number[], topK: number = 3): Promise<string[]> {
        try {
            const searchResponse = await this.client.search({
                index: this.indexName,
                body: {
                    size: topK,
                    query: {
                        knn: {
                            embedding: {
                                vector: queryEmbedding,
                                k: topK
                            }
                        }
                    },
                    _source: ['document']
                }
            });
            
            // Extract documents from search results
            const hits = searchResponse.body.hits.hits;
            return hits.map((hit: any) => hit._source.document);
        } catch (error) {
            console.error('Error searching in OpenSearch:', error);
            return [];
        }
    }

    async close(): Promise<void> {
        try {
            await this.client.close();
            console.log('OpenSearch connection closed');
        } catch (error) {
            console.error('Error closing OpenSearch connection:', error);
        }
    }
}
