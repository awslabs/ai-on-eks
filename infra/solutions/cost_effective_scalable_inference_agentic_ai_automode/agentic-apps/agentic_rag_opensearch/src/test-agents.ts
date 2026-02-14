import 'dotenv/config';
import { KnowledgeAgent, RAGAgent, MCPAgent, SupervisorAgent } from "./agents";
import MCPClient from "./MCPClient";
import { logTitle } from "./utils";
import path from "path";

// Simple test to verify all agents can be initialized
async function testAgents() {
    logTitle('TESTING MULTI-AGENT SYSTEM');
    
    try {
        // Test Knowledge Agent
        console.log('Testing Knowledge Agent...');
        const knowledgeAgent = new KnowledgeAgent();
        await knowledgeAgent.init();
        const hasChanges = await knowledgeAgent.checkForChanges();
        console.log(`Knowledge changes detected: ${hasChanges}`);
        await knowledgeAgent.close();
        console.log('✓ Knowledge Agent test passed');
        
        // Test RAG Agent
        console.log('\nTesting RAG Agent...');
        const ragAgent = new RAGAgent();
        await ragAgent.init();
        const stats = ragAgent.getStats();
        console.log('RAG Agent stats:', stats);
        await ragAgent.close();
        console.log('✓ RAG Agent test passed');
        
        // Test MCP Agent
        console.log('\nTesting MCP Agent...');
        const outPath = path.resolve(process.cwd(), 'output');
        const fileMCP = new MCPClient("filesystem", "npx", ['-y', '@modelcontextprotocol/server-filesystem', outPath]);
        const mcpAgent = new MCPAgent([fileMCP]);
        await mcpAgent.init();
        const mcpStats = mcpAgent.getStats();
        console.log('MCP Agent stats:', mcpStats);
        const tools = await mcpAgent.listAvailableTools();
        console.log(`Available tools: ${tools.length}`);
        await mcpAgent.close();
        console.log('✓ MCP Agent test passed');
        
        // Test Supervisor Agent
        console.log('\nTesting Supervisor Agent...');
        const fileMCP2 = new MCPClient("filesystem", "npx", ['-y', '@modelcontextprotocol/server-filesystem', outPath]);
        const supervisor = new SupervisorAgent([fileMCP2]);
        await supervisor.init();
        console.log('Supervisor Agent initialized successfully');
        await supervisor.close();
        console.log('✓ Supervisor Agent test passed');
        
        console.log('\n🎉 All agent tests passed successfully!');
        
    } catch (error) {
        console.error('❌ Agent test failed:', error);
        process.exit(1);
    }
}

// Run the test
(async () => {
    await testAgents();
})();
