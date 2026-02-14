import fs from "fs";
import path from "path";
import EmbeddingRetriever from "./EmbeddingRetriever";
import { logTitle } from "./utils";

// Function to process and embed knowledge files
async function embedKnowledgeFiles() {
    logTitle('EMBEDDING KNOWLEDGE FILES');
    
    const knowledgePath = path.join(process.cwd(), '..', 'knowledge');
    
    if (!fs.existsSync(knowledgePath)) {
        console.error(`Knowledge directory not found: ${knowledgePath}`);
        return false;
    }
    
    try {
        // Get all markdown files in the knowledge directory
        const files = fs.readdirSync(knowledgePath)
            .filter(file => file.endsWith('.md'));
        
        console.log(`Found ${files.length} markdown files in the knowledge directory`);
        
        // Initialize the embedding retriever with llamacpp-embedding model
        const embeddingRetriever = new EmbeddingRetriever("llamacpp-embedding");
        
        // Process each file
        for (const file of files) {
            const filePath = path.join(knowledgePath, file);
            console.log(`Processing file: ${file}`);
            
            // Read the file content
            const content = fs.readFileSync(filePath, 'utf-8');
            
            // Embed the document
            await embeddingRetriever.embedDocument(content);
        }
        
        console.log(`Successfully embedded ${files.length} knowledge files`);
        
        // Close OpenSearch connection when done
        // @ts-ignore - Access private property for cleanup
        if (embeddingRetriever.vectorStore && typeof embeddingRetriever.vectorStore.close === 'function') {
            // @ts-ignore
            await embeddingRetriever.vectorStore.close();
        }
        
        return true;
    } catch (error) {
        console.error("Error embedding knowledge files:", error);
        return false;
    }
}

// Main function
(async () => {
    const success = await embedKnowledgeFiles();
    
    if (success) {
        console.log("Knowledge embedding completed successfully");
    } else {
        console.error("Knowledge embedding failed");
        process.exit(1);
    }
})();
