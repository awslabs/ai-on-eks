import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import EmbeddingRetriever from "./EmbeddingRetriever";
import { logTitle } from "./utils";

// Function to update the RAG process with CSV data
async function updateRAGWithCSV() {
    logTitle('UPDATING RAG WITH CSV DATA');
    
    const csvFilePath = path.join(process.cwd(), '..', 'knowledge', 'q_c_data.csv');
    
    if (!fs.existsSync(csvFilePath)) {
        console.error(`File not found: ${csvFilePath}`);
        return false;
    }
    
    try {
        // Read the CSV file
        const fileContent = fs.readFileSync(csvFilePath, 'utf-8');
        
        // Parse the CSV content
        const records = parse(fileContent, {
            columns: true,
            skip_empty_lines: true
        });
        
        console.log(`Found ${records.length} records in the CSV file`);
        
        // Initialize the embedding retriever
        const embeddingRetriever = new EmbeddingRetriever("custom-embedding-model");
        
        // Process each record in batches to avoid overwhelming the system
        const batchSize = 50;
        let processedCount = 0;
        
        for (let i = 0; i < records.length; i += batchSize) {
            const batch = records.slice(i, i + batchSize);
            
            // Process batch in parallel
            await Promise.all(batch.map(async (record) => {
                // Combine question and context for better retrieval
                const documentText = `Question: ${record.question}\nContext: ${record.context}`;
                
                // Embed the document
                await embeddingRetriever.embedDocument(documentText);
            }));
            
            processedCount += batch.length;
            console.log(`Processed ${processedCount}/${records.length} records`);
        }
        
        console.log(`Successfully embedded ${processedCount} records from the CSV file`);
        
        // Close Milvus connection when done
        // @ts-ignore - Access private property for cleanup
        if (embeddingRetriever.vectorStore && typeof embeddingRetriever.vectorStore.close === 'function') {
            // @ts-ignore
            await embeddingRetriever.vectorStore.close();
        }
        
        return true;
    } catch (error) {
        console.error("Error updating RAG with CSV data:", error);
        return false;
    }
}

// Main function
(async () => {
    const success = await updateRAGWithCSV();
    
    if (success) {
        console.log("RAG update completed successfully");
    } else {
        console.error("RAG update failed");
        process.exit(1);
    }
})();
