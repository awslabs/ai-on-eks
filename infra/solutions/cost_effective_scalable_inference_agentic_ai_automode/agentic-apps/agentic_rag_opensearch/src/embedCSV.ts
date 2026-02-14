import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import EmbeddingRetriever from "./EmbeddingRetriever";
import { logTitle } from "./utils";

// Function to process and embed CSV data
async function processCSVFile(filePath: string) {
    logTitle('PROCESSING CSV FILE');
    console.log(`Processing file: ${filePath}`);
    
    try {
        // Read the CSV file
        const fileContent = fs.readFileSync(filePath, 'utf-8');
        
        // Parse the CSV content
        const records = parse(fileContent, {
            columns: true,
            skip_empty_lines: true
        });
        
        console.log(`Found ${records.length} records in the CSV file`);
        
        // Initialize the embedding retriever with llamacpp-embedding model
        const embeddingRetriever = new EmbeddingRetriever("llamacpp-embedding");
        
        // Process each record
        let processedCount = 0;
        for (const record of records) {
            // Combine question and context for better retrieval
            const documentText = `Question: ${record.question}\nContext: ${record.context}`;
            
            // Embed the document
            await embeddingRetriever.embedDocument(documentText);
            
            processedCount++;
            if (processedCount % 10 === 0) {
                console.log(`Processed ${processedCount}/${records.length} records`);
            }
        }
        
        console.log(`Successfully embedded ${processedCount} records from the CSV file`);
        
        // Close OpenSearch connection when done
        // @ts-ignore - Access private property for cleanup
        if (embeddingRetriever.vectorStore && typeof embeddingRetriever.vectorStore.close === 'function') {
            // @ts-ignore
            await embeddingRetriever.vectorStore.close();
        }
        
        return true;
    } catch (error) {
        console.error("Error processing CSV file:", error);
        return false;
    }
}

// Main function
(async () => {
    const csvFilePath = path.join(process.cwd(), '..', 'knowledge', 'q_c_data.csv');
    
    if (!fs.existsSync(csvFilePath)) {
        console.error(`File not found: ${csvFilePath}`);
        process.exit(1);
    }
    
    const success = await processCSVFile(csvFilePath);
    
    if (success) {
        console.log("CSV processing completed successfully");
    } else {
        console.error("CSV processing failed");
        process.exit(1);
    }
})();
