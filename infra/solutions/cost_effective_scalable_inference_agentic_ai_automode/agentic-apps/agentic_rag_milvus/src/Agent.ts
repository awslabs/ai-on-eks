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
            let clientName = name.split('___')[0];
            let toolName = name.split('___')[1] || name;
            
            // If no client name in the tool name, try to find a client that has this tool
            if (!toolName || toolName === name) {
                console.log(`No client prefix found in tool name: ${name}. Trying to find a matching client...`);
                toolName = name;
                
                // Try to find a client with this tool
                for (const client of this.mcpClients) {
                    const hasMatchingTool = client.getTools().some(tool => 
                        tool.name === name || tool.name.endsWith(`___${name}`)
                    );
                    
                    if (hasMatchingTool) {
                        clientName = client.mcp.name;
                        console.log(`Found matching client: ${clientName} for tool: ${name}`);
                        break;
                    }
                }
            }
            
            // Find the appropriate client
            const client = this.mcpClients.find(c => c.mcp.name === clientName);
            
            if (!client) {
                throw new Error(`No MCP client found for tool: ${name} (client: ${clientName}, tool: ${toolName})`);
            }
            
            console.log(`Using client: ${clientName} to call tool: ${toolName}`);
            
            // Special handling for write_file tool
            if (toolName === 'write_file') {
                try {
                    // Call the tool and get the result
                    const result = await client.callTool(toolName, args);
                    console.log(`Tool result: ${JSON.stringify(result).substring(0, 100)}...`);
                    
                    // Append the tool result to the conversation
                    this.llm?.appendToolResult(id, JSON.stringify(result));
                } catch (mcpError) {
                    console.error(`MCP tool call failed: ${mcpError}`);
                    
                    // Try fallback direct file writing
                    console.log("Attempting fallback file writing...");
                    try {
                        const fs = await import('fs');
                        const path = await import('path');
                        
                        // Get the output directory from the first client's args
                        const outPath = this.mcpClients[0]?.args[2] || process.cwd();
                        
                        // Construct the file path
                        const filePath = path.join(outPath, args.path);
                        
                        // Write the file
                        fs.writeFileSync(filePath, args.content);
                        
                        const result = { success: true, path: filePath };
                        console.log(`Fallback file writing succeeded: ${filePath}`);
                        
                        // Append the success result to the conversation
                        this.llm?.appendToolResult(id, JSON.stringify(result));
                    } catch (fsError) {
                        console.error(`Fallback file writing failed: ${fsError}`);
                        
                        // Append the error as the tool result
                        this.llm?.appendToolResult(id, JSON.stringify({ 
                            error: `Both MCP and fallback file writing failed: ${fsError.message}` 
                        }));
                    }
                }
            } else {
                // For other tools, proceed normally
                try {
                    // Call the tool and get the result
                    const result = await client.callTool(toolName, args);
                    console.log(`Tool result: ${JSON.stringify(result).substring(0, 100)}...`);
                    
                    // Append the tool result to the conversation
                    this.llm?.appendToolResult(id, JSON.stringify(result));
                } catch (error) {
                    console.error(`Error calling tool: ${error}`);
                    
                    // Append the error as the tool result
                    this.llm?.appendToolResult(id, JSON.stringify({ error: error.message }));
                }
            }
        } catch (error) {
            console.error(`Error processing tool call: ${error}`);
            console.error(`Error details:`, error);
            // Append the error as the tool result
            this.llm?.appendToolResult(toolCall.id, JSON.stringify({ error: error.message }));
        }
    }
}