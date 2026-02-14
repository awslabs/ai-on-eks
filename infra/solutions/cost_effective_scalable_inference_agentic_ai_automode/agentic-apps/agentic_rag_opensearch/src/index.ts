import 'dotenv/config';
import MCPClient from "./MCPClient";
import { SupervisorAgent } from "./agents";
import path from "path";
import fs from "fs";
import { logTitle } from "./utils";

// Verify environment variables are loaded
if (!process.env.AWS_REGION || !process.env.OPENSEARCH_ENDPOINT) {
  throw new Error('Required environment variables AWS_REGION and OPENSEARCH_ENDPOINT are not set');
}

// Use the parent directory (where the command is run) instead of src directory
const outPath = path.resolve(process.cwd(), 'output');
const TASK = `
Find information about "What is the most important aspect of initial treatment for Bell's palsy?". 
Summarize this information and create a comprehensive story about Bell's palsy treatment.
Save the story and important information to a file named "bells_palsy_treatment.md" in the output directory as a beautiful markdown file.
Include sections for:
1. Overview of Bell's palsy
2. Most important initial treatment aspects
3. Timeline for treatment
4. Expected outcomes
5. Additional recommendations
`

// Make sure output directory exists
if (!fs.existsSync(outPath)) {
  fs.mkdirSync(outPath, { recursive: true });
}

// Start the multi-agent application
(async () => {
  try {
    logTitle('INITIALIZING MULTI-AGENT RAG SYSTEM');
    
    // Initialize the filesystem MCP client
    const fileMCP = new MCPClient("filesystem", "npx", ['-y', '@modelcontextprotocol/server-filesystem', outPath]);
    
    await main(fileMCP);
  } catch (error) {
    console.error("Error in main:", error);
    process.exit(1);
  }
})();

async function main(fileMCP: MCPClient) {
  let supervisor: SupervisorAgent | null = null;
  
  try {
    // Initialize the supervisor agent with MCP clients
    logTitle('INITIALIZING SUPERVISOR AGENT');
    supervisor = new SupervisorAgent([fileMCP], 'Qwen/QwQ-32B-AWQ');
    await supervisor.init();
    
    // Execute the complete workflow
    logTitle('EXECUTING MULTI-AGENT WORKFLOW');
    console.log('Task:', TASK);
    
    const result = await supervisor.executeWorkflow(TASK);
    
    // Display results
    logTitle('WORKFLOW COMPLETED');
    console.log('Final Result:', result);
    
    // Display workflow summary
    const summary = supervisor.getWorkflowSummary();
    console.log('\n' + summary);
    
    // Display detailed task results
    const taskResults = supervisor.getTaskResults();
    console.log('\nDetailed Task Results:');
    for (const [taskId, result] of taskResults) {
      console.log(`- ${taskId}: ${result.success ? 'SUCCESS' : 'FAILED'}`);
      if (result.error) {
        console.log(`  Error: ${result.error}`);
      }
    }
    
  } catch (error) {
    console.error("Error in workflow execution:", error);
    throw error;
  } finally {
    // Clean up resources
    if (supervisor) {
      logTitle('CLEANING UP');
      await supervisor.close();
    }
  }
}
