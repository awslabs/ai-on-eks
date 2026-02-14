import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { Tool } from "@modelcontextprotocol/sdk/types.js";

export default class MCPClient {
    public mcp: Client;
    private command: string;
    private args: string[]
    private transport: StdioClientTransport | null = null;
    private tools: Tool[] = [];

    constructor(name: string, command: string, args: string[], version?: string) {
        this.mcp = new Client({ name, version: version || "0.0.1" });
        this.command = command;
        this.args = args;
    }

    public async init() {
        await this.connectToServer();
    }

    public async close() {
        await this.mcp.close();
    }

    public getTools() {
        return this.tools;
    }

    public callTool(name: string, params: Record<string, any>) {
        return this.mcp.callTool({
            name,
            arguments: params,
        });
    }

    private async connectToServer() {
        try {
            this.transport = new StdioClientTransport({
                command: this.command,
                args: this.args,
            });
            await this.mcp.connect(this.transport);

            try {
                const toolsResult = await this.mcp.listTools();
                this.tools = toolsResult.tools.map((tool) => {
                    // Prefix tool names with the client name to ensure proper routing
                    const prefixedName = `${this.mcp.name}___${tool.name}`;
                    return {
                        name: prefixedName,
                        description: tool.description,
                        inputSchema: tool.inputSchema,
                    };
                });
                console.log(
                    "Connected to server with tools:",
                    this.tools.map(({ name }) => name)
                );
            } catch (toolError) {
                console.error("Failed to list tools from MCP server: ", toolError);
                console.log("Adding fallback write_file tool manually");
                
                // Add a fallback write_file tool manually
                this.tools = [{
                    name: `${this.mcp.name}___write_file`,
                    description: "Write content to a file",
                    inputSchema: {
                        type: "object",
                        properties: {
                            path: { type: "string", description: "Path to the file" },
                            content: { type: "string", description: "Content to write" }
                        },
                        required: ["path", "content"]
                    }
                }];
            }
        } catch (e) {
            console.error("Failed to connect to MCP server: ", e);
            console.error("Error details:", e);
            console.log("Will continue without MCP tools and rely on fallback methods");
            
            // Add a dummy tool so the agent can still make tool calls
            this.tools = [{
                name: `${this.mcp.name}___write_file`,
                description: "Write content to a file (fallback)",
                inputSchema: {
                    type: "object",
                    properties: {
                        path: { type: "string", description: "Path to the file" },
                        content: { type: "string", description: "Content to write" }
                    },
                    required: ["path", "content"]
                }
            }];
        }
    }
}