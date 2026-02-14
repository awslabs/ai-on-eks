import MCPClient from "./MCPClient";
import Agent from "./Agent";
import path from "path";
import EmbeddingRetriever from "./EmbeddingRetriever";
import fs from "fs";
import { logTitle } from "./utils";

// Use the parent directory (where the command is run) instead of src directory
const outPath = path.resolve(process.cwd(), 'output');
const TASK = `
Give me the answer about What is the primary advantage of using an uncemented porous-coated anatomic (PCA) hip system in total hip arthroplasty? 
Summarize this information and create a story about it.
Save the story to a file named "Bell's palsy.md" in the output directory as a beautiful markdown file.
`

// Make sure output directory exists
if (!fs.existsSync(outPath)) {
  fs.mkdirSync(outPath, { recursive: true });
}

// Add test function for Milvus search
async function testMilvusSearch() {
  console.log('Testing Milvus search functionality...');
  
  const embeddingRetriever = new EmbeddingRetriever("custom-embedding-model");
  
  // Generate a test embedding (random values)
  const testEmbedding = Array(1536).fill(0).map(() => Math.random());
  
  // Try a direct search with the test embedding
  // @ts-ignore - Access private property for testing
  const results = await embeddingRetriever.vectorStore.search(testEmbedding, 5);
  
  console.log(`Test search returned ${results.length} results`);
  if (results.length > 0) {
    console.log('First result preview:', results[0].substring(0, 100) + '...');
  } else {
    console.log('No results found in test search');
  }
  
  // Try a simple query that should match something
  const simpleQuery = "surgical complications";
  console.log(`Testing simple query: "${simpleQuery}"`);
  const queryResults = await embeddingRetriever.retrieve(simpleQuery, 5);
  
  console.log(`Simple query returned ${queryResults.length} results`);
  if (queryResults.length > 0) {
    console.log('First result preview:', queryResults[0].substring(0, 100) + '...');
  } else {
    console.log('No results found for simple query');
  }
}

// Start the application immediately
(async () => {
  try {
    logTitle('INITIALIZING AGENTIC RAG SYSTEM');
    
    // Run test search first
    await testMilvusSearch();
    
    // Initialize the filesystem MCP client
    const fileMCP = new MCPClient("mcp-server-file", "npx", ['-y', '@modelcontextprotocol/server-filesystem', outPath]);
    
    // Add a fallback method to write files directly if MCP fails
    const fallbackWriteFile = async (filename: string, content: string) => {
      try {
        const filePath = path.join(outPath, filename);
        fs.writeFileSync(filePath, content);
        console.log(`Fallback: Successfully wrote file to ${filePath}`);
        return { success: true, path: filePath };
      } catch (error) {
        console.error(`Fallback: Failed to write file: ${error}`);
        return { success: false, error: error.message };
      }
    };
    
    await main(fileMCP, fallbackWriteFile);
  } catch (error) {
    console.error("Error in main:", error);
  }
})();

async function main(fileMCP: MCPClient, fallbackWriteFile?: (filename: string, content: string) => Promise<any>) {
  // Step 1: Retrieve relevant context using RAG
  const context = await retrieveContext("What is the primary advantage of using an uncemented porous-coated anatomic (PCA) hip system in total hip arthroplasty?");
  
  // Step 2: Initialize the agent with the context and MCP client
  logTitle('INITIALIZING AGENT');
  const systemPrompt = `You are a helpful assistant that can retrieve information and create stories.
You have access to tools that can help you complete tasks.
When asked to save files, always use the filesystem tool to write the content.
Specifically, use the mcp-server-file___write_file tool to save files.
The output path is ${outPath}.`;

  const agent = new Agent('Qwen/QwQ-32B-AWQ', [fileMCP], systemPrompt, context);
  await agent.init();
  
  // Step 3: Execute the task with the agent
  logTitle('EXECUTING TASK');
  console.log(TASK);
  
  try {
    const response = await agent.invoke(TASK);
    
    // Step 4: Close connections
    logTitle('TASK COMPLETED');
    console.log(response);
    
    // Always use the fallback method to save the file
    logTitle('USING FALLBACK METHOD TO SAVE FILE');
    
    // Extract the content from the response - try different patterns
    let content = null;
    
    // Try to extract markdown content
    const markdownMatch = response.match(/```markdown\n([\s\S]*?)\n```/);
    if (markdownMatch && markdownMatch[1]) {
      content = markdownMatch[1];
    } else {
      // Try to extract any code block content
      const codeBlockMatch = response.match(/```(?:\w+)?\n([\s\S]*?)\n```/);
      if (codeBlockMatch && codeBlockMatch[1]) {
        content = codeBlockMatch[1];
      } else {
        // If no code blocks, use the entire response
        content = response;
      }
    }
    
    if (content && fallbackWriteFile) {
      console.log("Extracted content for file writing:", content.substring(0, 100) + "...");
      const result = await fallbackWriteFile("antonette.md", content);
      if (result.success) {
        console.log(`Successfully saved file using fallback method to ${result.path}`);
      } else {
        console.error(`Failed to save file using fallback method: ${result.error}`);
      }
    } else {
      console.error("Could not extract content from response or fallback method not available");
    }
  } catch (error) {
    console.error("Error during task execution:", error);
  } finally {
    await agent.close();
    
    // Close Milvus connection when done
    const embeddingRetriever = new EmbeddingRetriever("custom-embedding-model");
    // @ts-ignore - Access private property for cleanup
    if (embeddingRetriever.vectorStore && typeof embeddingRetriever.vectorStore.close === 'function') {
      // @ts-ignore
      await embeddingRetriever.vectorStore.close();
    }
  }
}

async function retrieveContext(query: string) {
    logTitle('RETRIEVING CONTEXT');
    console.log(`Query: ${query}`);
    
    // Initialize the embedding retriever
    const embeddingRetriever = new EmbeddingRetriever("custom-embedding-model");
    
    // Retrieve relevant documents
    const documents = await embeddingRetriever.retrieve(query, 5);
    
    // Combine documents into context
    const context = documents.join('\n\n');
    
    console.log(`Retrieved ${documents.length} relevant documents`);
    console.log('Context preview:', context.substring(0, 200) + '...');
    
    return context;
}
