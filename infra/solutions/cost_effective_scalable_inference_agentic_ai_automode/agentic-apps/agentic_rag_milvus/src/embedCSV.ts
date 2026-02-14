import fs from "fs";
import path from "path";
import { parse } from "csv-parse/sync";
import EmbeddingRetriever from "./EmbeddingRetriever";
import { logTitle } from "./utils";

// Function to process and embed CSV data
async function embedCSVData() {
    logTitle('EMBEDDING CSV DATA');
    
    const csvPath = path.join(process.cwd(), 'knowledge');
    
    if (!fs.existsSync(csvPath)) {
        console.error(`CSV data directory not found: ${csvPath}`);
        return false;
    }
    
    try {
        // Get all CSV files in the data directory
        const files = fs.readdirSync(csvPath)
            .filter(file => file.endsWith('.csv'));
        
        console.log(`Found ${files.length} CSV files in the data directory`);
        
        // Initialize the embedding retriever
        const embeddingRetriever = new EmbeddingRetriever("custom-embedding-model");
        
        // Process each file
        for (const file of files) {
            const filePath = path.join(csvPath, file);
            console.log(`Processing file: ${file}`);
            
            // Read the file content
            const content = fs.readFileSync(filePath, 'utf-8');
            
            // Parse CSV
            const records = parse(content, {
                columns: true,
                skip_empty_lines: true
            });
            
            console.log(`Found ${records.length} records in ${file}`);
            
            // Process each record
            for (const record of records) {
                // Convert record to a string representation
                const recordString = Object.entries(record)
                    .map(([key, value]) => `${key}: ${value}`)
                    .join('\n');
                
                // Create a document with metadata
                const document = `# ${record.name || record.title || record.id || 'Record'}\n\n${recordString}`;
                
                // Embed the document
                await embeddingRetriever.embedDocument(document);
            }
            
            console.log(`Successfully embedded ${records.length} records from ${file}`);
        }
        
        // Close Milvus connection when done
        // @ts-ignore - Access private property for cleanup
        if (embeddingRetriever.vectorStore && typeof embeddingRetriever.vectorStore.close === 'function') {
            // @ts-ignore
            await embeddingRetriever.vectorStore.close();
        }
        
        return true;
    } catch (error) {
        console.error("Error embedding CSV data:", error);
        return false;
    }
}

// Main function
(async () => {
    const success = await embedCSVData();
    
    if (success) {
        console.log("CSV data embedding completed successfully");
    } else {
        console.error("CSV data embedding failed");
        process.exit(1);
    }
})();
