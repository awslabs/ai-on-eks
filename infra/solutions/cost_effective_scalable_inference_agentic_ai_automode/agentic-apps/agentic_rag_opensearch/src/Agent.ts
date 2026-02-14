import MCPClient from "./MCPClient";
import ChatOpenAI from "./ChatOpenAI";
import { logTitle } from "./utils";
import { ToolCall } from "./ChatOpenAI";

export default class Agent {
    private mcpClients: MCPClient[];
    private llm: ChatOpenAI | null = null;
    private model: string;
    private systemPrompt: string;
    private context: string;

    constructor(model: string, mcpClients: MCPClient[], systemPrompt: string = '', context: string = '') {
        this.mcpClients = mcpClients;
        this.model = model;
        this.systemPrompt = systemPrompt;
        this.context = context;
    }

    async init() {
        logTitle('TOOLS');
        for await (const client of this.mcpClients) {
            await client.init();
        }
        const tools = this.mcpClients.flatMap(client => client.getTools());
        this.llm = new ChatOpenAI(this.model, this.systemPrompt, tools, this.context);
    }

    async close() {
        for await (const client of this.mcpClients) {
            await client.close();
        }
    }

    async invoke(prompt: string) {
        if (!this.llm) throw new Error('Agent not initialized');
        
        try {
            logTitle('AGENT EXECUTION');
            console.log("Invoking LLM with tools...");
            
            // Start the conversation with the user prompt
            let response = await this.llm.chat(prompt);
            
            // Continue the conversation until no more tool calls are needed
            while (response.toolCalls && response.toolCalls.length > 0) {
                logTitle('TOOL CALLS');
                console.log(`Processing ${response.toolCalls.length} tool calls`);
                
                // Process each tool call
                for (const toolCall of response.toolCalls) {
                    await this.processToolCall(toolCall);
                }
                
                // Continue the conversation with the tool results
                response = await this.llm.chat();
            }
            
            logTitle('FINAL RESPONSE');
            console.log("Successfully completed request");
            return response.content;
        } catch (error) {
            console.error("Error in agent execution:", error);
            throw error;
        }
    }
    
    private async processToolCall(toolCall: ToolCall) {
        try {
            const { id, function: { name, arguments: argsString } } = toolCall;
            console.log(`Executing tool call: ${name}`);
            
            // Parse the arguments
            const args = JSON.parse(argsString);
            
            // Find the MCP client that can handle this tool
            const toolName = name;
            
            // Find the appropriate client
            const client = this.mcpClients[0]; // Since we only have one client
            
            if (!client) {
                throw new Error(`No MCP client found for tool: ${name}`);
            }
            
            // Call the tool and get the result
            const result = await client.callTool(toolName, args);
            console.log(`Tool result: ${JSON.stringify(result).substring(0, 100)}...`);
            
            // Append the tool result to the conversation
            this.llm?.appendToolResult(id, JSON.stringify(result));
        } catch (error) {
            console.error(`Error processing tool call: ${error}`);
            // Append the error as the tool result
            this.llm?.appendToolResult(toolCall.id, JSON.stringify({ error: error.message }));
        }
    }
}