import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node';
import { logTitle } from "./utils";
import 'dotenv/config';

// Function to rebuild the Milvus collection with 384 dimensions
async function rebuildCollection() {
    logTitle('REBUILDING MILVUS COLLECTION');
    
    const collectionName = 'rag_documents_384d';
    const dimension = 384; // New dimension to match the custom embedding endpoint
    
    try {
        // Connect to Milvus
        const client = new MilvusClient({
            address: process.env.MILVUS_ADDRESS || '',
            username: process.env.MILVUS_USERNAME || '',
            password: process.env.MILVUS_PASSWORD || '',
        });
        
        // Check if collection exists and drop it if it does
        const hasCollection = await client.hasCollection({
            collection_name: collectionName,
        });
        
        if (hasCollection.value) {
            console.log(`Collection ${collectionName} already exists. Dropping it...`);
            await client.dropCollection({
                collection_name: collectionName,
            });
            console.log(`Collection ${collectionName} dropped successfully.`);
        }
        
        // Create new collection with 384 dimensions
        console.log(`Creating new collection ${collectionName} with ${dimension} dimensions...`);
        await client.createCollection({
            collection_name: collectionName,
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
                    dim: dimension,
                },
                {
                    name: 'document',
                    data_type: DataType.VarChar,
                    max_length: 65535,
                },
            ],
        });
        
        // Create index for vector search
        console.log(`Creating index for collection ${collectionName}...`);
        await client.createIndex({
            collection_name: collectionName,
            field_name: 'embedding',
            index_type: 'HNSW',
            metric_type: 'COSINE',
            params: { M: 8, efConstruction: 64 },
        });
        
        // Load collection into memory
        console.log(`Loading collection ${collectionName} into memory...`);
        await client.loadCollection({
            collection_name: collectionName,
        });
        
        console.log(`Collection ${collectionName} created and indexed successfully.`);
        
        // Close connection
        await client.closeConnection();
        
        return true;
    } catch (error) {
        console.error("Error rebuilding Milvus collection:", error);
        return false;
    }
}

// Main function
(async () => {
    const success = await rebuildCollection();
    
    if (success) {
        console.log("Collection rebuild completed successfully. Now you need to re-embed your documents.");
        console.log("Run 'pnpm embed-knowledge' and 'pnpm embed-csv' to populate the new collection.");
    } else {
        console.error("Collection rebuild failed");
        process.exit(1);
    }
})();
