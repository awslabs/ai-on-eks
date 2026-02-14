import { Langfuse } from 'langfuse';
import 'dotenv/config';

// Validate environment variables
const publicKey = process.env.LANGFUSE_PUBLIC_KEY;
const secretKey = process.env.LANGFUSE_SECRET_KEY;
const host = process.env.LANGFUSE_HOST;

if (!publicKey || !secretKey || !host) {
  console.warn('Langfuse configuration incomplete. Some environment variables are missing:');
  console.warn('- LANGFUSE_PUBLIC_KEY:', publicKey ? '✓' : '✗');
  console.warn('- LANGFUSE_SECRET_KEY:', secretKey ? '✓' : '✗');
  console.warn('- LANGFUSE_HOST:', host ? '✓' : '✗');
  console.warn('Langfuse tracing will be disabled.');
}

// Initialize Langfuse client
export const langfuse = publicKey && secretKey && host ? new Langfuse({
  secretKey,
  publicKey,
  baseUrl: host,
  flushAt: 1, // Send traces immediately for development
}) : null;

// Helper function to create a trace
export function createTrace(name: string, input?: any, metadata?: any) {
  if (!langfuse) {
    console.warn('Langfuse not configured, skipping trace creation');
    return null;
  }
  
  return langfuse.trace({
    name,
    input,
    metadata: {
      ...metadata,
      timestamp: new Date().toISOString(),
      environment: 'development'
    }
  });
}

// Helper function to create a span within a trace
export function createSpan(trace: any, name: string, input?: any, metadata?: any) {
  if (!trace) {
    return null;
  }
  
  return trace.span({
    name,
    input,
    metadata: {
      ...metadata,
      timestamp: new Date().toISOString()
    }
  });
}

// Helper function to create a generation (LLM call) within a trace
export function createGeneration(trace: any, name: string, input?: any, model?: string, metadata?: any) {
  if (!trace) {
    return null;
  }
  
  return trace.generation({
    name,
    input,
    model,
    metadata: {
      ...metadata,
      timestamp: new Date().toISOString()
    }
  });
}

// Helper function to flush traces (useful for cleanup)
export async function flushLangfuse() {
  if (langfuse) {
    await langfuse.flushAsync();
  }
}

// Export configuration status
export const isLangfuseEnabled = !!langfuse;

console.log(`Langfuse tracing: ${isLangfuseEnabled ? 'ENABLED' : 'DISABLED'}`);
