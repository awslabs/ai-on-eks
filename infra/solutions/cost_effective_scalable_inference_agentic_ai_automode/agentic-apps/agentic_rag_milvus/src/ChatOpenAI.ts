import OpenAI from "openai";
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import 'dotenv/config'
import { logTitle } from "./utils";

export interface ToolCall {
    id: string;
    function: {
        name: string;
        arguments: string;
    };
}

export default class ChatOpenAI {
    private llm: OpenAI;
    private model: string;
    private messages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
    private tools: Tool[];

    constructor(model: string, systemPrompt: string = '', tools: Tool[] = [], context: string = '') {
        this.llm = new OpenAI({
            apiKey: process.env.OPENAI_API_KEY,
            baseURL: process.env.OPENAI_BASE_URL,
            defaultHeaders: {
                // Use only one authentication method to avoid conflicts
                "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
                // Removed "api-key" header to prevent double authentication
            }
        });
        this.model = model;
        this.tools = tools;
        if (systemPrompt) this.messages.push({ role: "system", content: systemPrompt });
        if (context) this.messages.push({ role: "user", content: `Here is some context that might be helpful: \n\n${context}` });
    }

    async chat(prompt?: string): Promise<{ content: string, toolCalls: ToolCall[] }> {
        logTitle('CHAT');
        if (prompt) {
            this.messages.push({ role: "user", content: prompt });
        }
        
        // Debug the request
        console.log('Sending request to model:', this.model);
        console.log('Messages count:', this.messages.length);
        console.log('Tools count:', this.tools.length);
        
        try {
            // Create request options with tools if available
            const requestOptions: OpenAI.Chat.ChatCompletionCreateParams = {
                model: this.model,
                messages: this.messages,
                stream: false,
            };
            
            // Add tools if available
            if (this.tools.length > 0) {
                requestOptions.tools = this.getToolsDefinition();
                requestOptions.tool_choice = "auto";
            }
            
            console.log('Sending request with options:', JSON.stringify({
                ...requestOptions,
                messages: `[${requestOptions.messages.length} messages]` // Don't log full messages
            }, null, 2));
            
            const completion = await this.llm.chat.completions.create(requestOptions);
            
            // Extract the response content and tool calls
            const message = completion.choices[0]?.message;
            const content = message?.content || "";
            const toolCalls = message?.tool_calls?.map(tc => ({
                id: tc.id,
                function: {
                    name: tc.function.name,
                    arguments: tc.function.arguments
                }
            })) || [];
            
            // Add the response to messages
            this.messages.push({
                role: "assistant",
                content: content,
                tool_calls: message?.tool_calls
            });
            
            // Log the response
            if (toolCalls.length > 0) {
                console.log(`Response includes ${toolCalls.length} tool calls`);
                toolCalls.forEach(tc => console.log(`- Tool: ${tc.function.name}`));
            } else {
                console.log('Response content:', content.substring(0, 100) + (content.length > 100 ? '...' : ''));
            }
            
            return {
                content: content,
                toolCalls: toolCalls,
            };
        } catch (error) {
            console.error('Error details:', error);
            
            // Enhanced error logging
            console.error('Full error object:', JSON.stringify(error, null, 2));
            
            if (error.response) {
                console.error('Response status:', error.response.status);
                console.error('Response headers:', error.response.headers);
                try {
                    const responseText = await error.response.text();
                    console.error('Response data:', responseText);
                    try {
                        // Try to parse as JSON for better readability
                        const responseJson = JSON.parse(responseText);
                        console.error('Response JSON:', JSON.stringify(responseJson, null, 2));
                    } catch (parseError) {
                        // If not valid JSON, the text version is already logged
                    }
                } catch (e) {
                    console.error('Could not read response data:', e);
                }
            }
            
            // Log the exact request that was sent
            console.error('Request that caused the error:');
            console.error('- URL:', process.env.OPENAI_BASE_URL + '/chat/completions');
            console.error('- Model:', this.model);
            console.error('- Messages count:', this.messages.length);
            console.error('- Tools count:', this.tools.length);
            
            throw error;
        }
    }

    public appendToolResult(toolCallId: string, toolOutput: string) {
        this.messages.push({
            role: "tool",
            content: toolOutput,
            tool_call_id: toolCallId
        });
    }

    private getToolsDefinition(): OpenAI.Chat.Completions.ChatCompletionTool[] {
        return this.tools.map((tool) => ({
            type: "function",
            function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.inputSchema,
            },
        }));
    }
}
