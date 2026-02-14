import 'dotenv/config';
import { KnowledgeAgent } from "../agents";
import { logTitle } from "../utils";

// Function to run knowledge embedding as a standalone script
async function runKnowledgeEmbedding() {
    logTitle('STANDALONE KNOWLEDGE EMBEDDING');
    
    let knowledgeAgent: KnowledgeAgent | null = null;
    
    try {
        // Initialize the knowledge agent
        knowledgeAgent = new KnowledgeAgent();
        await knowledgeAgent.init();
        
        // Check for changes and embed if needed
        const hasChanges = await knowledgeAgent.checkForChanges();
        
        if (hasChanges) {
            console.log('Changes detected, embedding all knowledge files (including CSV)...');
            
            // Embed all knowledge files (markdown, text, JSON, and CSV)
            const result = await knowledgeAgent.embedKnowledge();
            console.log(`Knowledge embedding result: ${result ? 'SUCCESS' : 'FAILED'}`);
            
            if (result) {
                console.log('Knowledge embedding completed successfully');
            } else {
                console.error('Knowledge embedding failed');
                process.exit(1);
            }
        } else {
            console.log('No changes detected in knowledge files');
        }
        
    } catch (error) {
        console.error('Error in knowledge embedding:', error);
        process.exit(1);
    } finally {
        // Clean up
        if (knowledgeAgent) {
            await knowledgeAgent.close();
        }
    }
}

// Main execution
(async () => {
    await runKnowledgeEmbedding();
})();
