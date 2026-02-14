import { logTitle } from "../utils";
import KnowledgeAgent from "./KnowledgeAgent";
import RAGAgent from "./RAGAgent";
import MCPAgent from "./MCPAgent";
import MCPClient from "../MCPClient";
import { createTrace, createSpan, flushLangfuse, isLangfuseEnabled } from "../LangfuseConfig";

export interface AgentTask {
    id: string;
    type: 'knowledge_check' | 'embedding' | 'rag_query' | 'mcp_tool';
    description: string;
    data: any;
}

export interface AgentResult {
    taskId: string;
    success: boolean;
    data?: any;
    error?: string;
}

export default class SupervisorAgent {
    private knowledgeAgent: KnowledgeAgent;
    private ragAgent: RAGAgent;
    private mcpAgent: MCPAgent;
    private taskQueue: AgentTask[] = [];
    private results: Map<string, AgentResult> = new Map();
    private trace: any; // Langfuse trace

    constructor(mcpClients: MCPClient[], model: string = 'Qwen/QwQ-32B-AWQ') {
        this.knowledgeAgent = new KnowledgeAgent();
        this.ragAgent = new RAGAgent(model);
        this.mcpAgent = new MCPAgent(mcpClients, model);
    }

    async init() {
        logTitle('INITIALIZING SUPERVISOR AGENT');
        console.log('Initializing all sub-agents...');
        
        await Promise.all([
            this.knowledgeAgent.init(),
            this.ragAgent.init(),
            this.mcpAgent.init()
        ]);
        
        console.log('All agents initialized successfully');
    }

    async close() {
        logTitle('CLOSING SUPERVISOR AGENT');
        await Promise.all([
            this.knowledgeAgent.close(),
            this.ragAgent.close(),
            this.mcpAgent.close()
        ]);
        
        // Flush Langfuse traces before closing
        if (isLangfuseEnabled) {
            console.log('Flushing Langfuse traces...');
            await flushLangfuse();
        }
    }

    async executeWorkflow(userQuery: string): Promise<string> {
        logTitle('EXECUTING MULTI-AGENT WORKFLOW');
        console.log(`User Query: ${userQuery}`);

        // Create main trace for the entire workflow
        this.trace = createTrace(
            'multi-agent-workflow',
            { userQuery },
            {
                agentType: 'SupervisorAgent',
                workflowType: 'multi-agent-rag',
                timestamp: new Date().toISOString()
            }
        );

        try {
            // Step 1: Check knowledge and update if needed
            const knowledgeSpan = createSpan(this.trace, 'knowledge-check', { step: 1 });
            const knowledgeStatus = await this.checkAndUpdateKnowledge();
            if (knowledgeSpan && isLangfuseEnabled) {
                knowledgeSpan.end({ output: { hasChanges: knowledgeStatus } });
            }
            
            // Step 2: Retrieve context using RAG
            const ragSpan = createSpan(this.trace, 'rag-retrieval', { step: 2, userQuery });
            const context = await this.retrieveContext(userQuery);
            if (ragSpan && isLangfuseEnabled) {
                ragSpan.end({ 
                    output: { 
                        contextLength: context.length,
                        contextPreview: context.substring(0, 200) + '...'
                    } 
                });
            }
            
            // Step 3: Execute task with MCP tools
            const mcpSpan = createSpan(this.trace, 'mcp-execution', { step: 3, userQuery, contextLength: context.length });
            const finalResult = await this.executeWithTools(userQuery, context);
            if (mcpSpan && isLangfuseEnabled) {
                mcpSpan.end({ 
                    output: { 
                        resultLength: finalResult.length,
                        resultPreview: finalResult.substring(0, 200) + '...'
                    } 
                });
            }

            // End main trace with success
            if (this.trace && isLangfuseEnabled) {
                this.trace.update({
                    output: {
                        success: true,
                        resultLength: finalResult.length,
                        tasksCompleted: this.results.size
                    }
                });
            }
            
            return finalResult;
        } catch (error) {
            console.error('Error in workflow execution:', error);
            
            // End main trace with error
            if (this.trace && isLangfuseEnabled) {
                this.trace.update({
                    output: {
                        success: false,
                        error: error.message || 'Unknown error',
                        errorType: error.constructor.name
                    }
                });
            }
            
            throw error;
        }
    }

    private async checkAndUpdateKnowledge(): Promise<boolean> {
        logTitle('STEP 1: KNOWLEDGE CHECK & UPDATE');
        
        const task: AgentTask = {
            id: 'knowledge-check-' + Date.now(),
            type: 'knowledge_check',
            description: 'Check for knowledge changes and update embeddings',
            data: {}
        };

        const hasChanges = await this.knowledgeAgent.checkForChanges();
        
        if (hasChanges) {
            console.log('Knowledge changes detected, updating embeddings...');
            const embeddingResult = await this.knowledgeAgent.embedKnowledge();
            
            this.results.set(task.id, {
                taskId: task.id,
                success: embeddingResult,
                data: { hasChanges, embeddingResult }
            });
            
            return embeddingResult;
        } else {
            console.log('No knowledge changes detected');
            this.results.set(task.id, {
                taskId: task.id,
                success: true,
                data: { hasChanges: false }
            });
            
            return true;
        }
    }

    private async retrieveContext(query: string): Promise<string> {
        logTitle('STEP 2: RAG CONTEXT RETRIEVAL');
        
        const task: AgentTask = {
            id: 'rag-query-' + Date.now(),
            type: 'rag_query',
            description: 'Retrieve relevant context using RAG',
            data: { query }
        };

        try {
            const context = await this.ragAgent.retrieveContext(query);
            
            this.results.set(task.id, {
                taskId: task.id,
                success: true,
                data: { context }
            });
            
            return context;
        } catch (error) {
            this.results.set(task.id, {
                taskId: task.id,
                success: false,
                error: error.message
            });
            
            throw error;
        }
    }

    private async executeWithTools(query: string, context: string): Promise<string> {
        logTitle('STEP 3: MCP TOOL EXECUTION');
        
        const task: AgentTask = {
            id: 'mcp-execution-' + Date.now(),
            type: 'mcp_tool',
            description: 'Execute task with MCP tools',
            data: { query, context }
        };

        try {
            const result = await this.mcpAgent.executeTask(query, context, this.trace);
            
            this.results.set(task.id, {
                taskId: task.id,
                success: true,
                data: { result }
            });
            
            return result;
        } catch (error) {
            this.results.set(task.id, {
                taskId: task.id,
                success: false,
                error: error.message
            });
            
            throw error;
        }
    }

    getTaskResults(): Map<string, AgentResult> {
        return this.results;
    }

    getWorkflowSummary(): string {
        const results = Array.from(this.results.values());
        const successful = results.filter(r => r.success).length;
        const failed = results.filter(r => !r.success).length;
        
        return `Workflow Summary: ${successful} successful tasks, ${failed} failed tasks`;
    }
}
