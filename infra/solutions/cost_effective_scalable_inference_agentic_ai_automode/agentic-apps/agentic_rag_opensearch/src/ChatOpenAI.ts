import OpenAI from "openai";
import { Tool } from "@modelcontextprotocol/sdk/types.js";
import 'dotenv/config'
import { logTitle } from "./utils";
import { createGeneration, isLangfuseEnabled } from "./LangfuseConfig";

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
    private trace: any; // Langfuse trace

    constructor(model: string, systemPrompt: string = '', tools: Tool[] = [], context: string = '', trace?: any) {
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
        this.trace = trace;
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
        
        // Create Langfuse generation for tracing
        const generation = createGeneration(
            this.trace,
            `chat-${this.model}`,
            {
                messages: this.messages.map(m => ({ role: m.role, content: typeof m.content === 'string' ? m.content.substring(0, 200) + '...' : '[complex content]' })),
                tools: this.tools.map(t => t.name),
                model: this.model
            },
            this.model,
            {
                messageCount: this.messages.length,
                toolCount: this.tools.length,
                hasSystemPrompt: this.messages.some(m => m.role === 'system')
            }
        );
        
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
            
            const startTime = Date.now();
            const completion = await this.llm.chat.completions.create(requestOptions);
            const endTime = Date.now();
            
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
            
            // Update Langfuse generation with results
            if (generation && isLangfuseEnabled) {
                generation.end({
                    output: {
                        content: content.substring(0, 500) + (content.length > 500 ? '...' : ''),
                        toolCalls: toolCalls.map(tc => ({ name: tc.function.name, id: tc.id }))
                    },
                    usage: {
                        promptTokens: completion.usage?.prompt_tokens,
                        completionTokens: completion.usage?.completion_tokens,
                        totalTokens: completion.usage?.total_tokens
                    },
                    metadata: {
                        duration: endTime - startTime,
                        finishReason: completion.choices[0]?.finish_reason,
                        model: completion.model
                    }
                });
            }
            
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
            // Update Langfuse generation with error
            if (generation && isLangfuseEnabled) {
                generation.end({
                    output: null,
                    metadata: {
                        error: error.message || 'Unknown error',
                        errorType: error.constructor.name
                    }
                });
            }
            
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

    public updateSystemPrompt(newSystemPrompt: string) {
        // Remove existing system message if present
        this.messages = this.messages.filter(msg => msg.role !== "system");
        
        // Add new system message at the beginning
        this.messages.unshift({ role: "system", content: newSystemPrompt });
    }

    public clearMessages() {
        this.messages = [];
    }

    public getMessageCount(): number {
        return this.messages.length;
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
