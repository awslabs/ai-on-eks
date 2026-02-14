import 'dotenv/config';
import { createTrace, createSpan, createGeneration, flushLangfuse, isLangfuseEnabled } from './LangfuseConfig';

async function testLangfuseIntegration() {
    console.log('Testing Langfuse Integration...');
    console.log('Langfuse enabled:', isLangfuseEnabled);
    
    if (!isLangfuseEnabled) {
        console.log('Langfuse is not enabled. Please check your environment variables.');
        return;
    }
    
    // Create a test trace
    const trace = createTrace(
        'test-trace',
        { message: 'Testing Langfuse integration' },
        { testType: 'integration-test' }
    );
    
    if (!trace) {
        console.log('Failed to create trace');
        return;
    }
    
    console.log('Created trace successfully');
    
    // Create a test span
    const span = createSpan(
        trace,
        'test-span',
        { operation: 'test-operation' },
        { spanType: 'test' }
    );
    
    if (span) {
        console.log('Created span successfully');
        
        // Simulate some work
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // End the span
        span.end({
            output: { result: 'test completed successfully' }
        });
        
        console.log('Ended span successfully');
    }
    
    // Create a test generation
    const generation = createGeneration(
        trace,
        'test-generation',
        { prompt: 'Test prompt' },
        'test-model',
        { generationType: 'test' }
    );
    
    if (generation) {
        console.log('Created generation successfully');
        
        // Simulate LLM response
        await new Promise(resolve => setTimeout(resolve, 200));
        
        // End the generation
        generation.end({
            output: { response: 'Test response from LLM' },
            usage: {
                promptTokens: 10,
                completionTokens: 5,
                totalTokens: 15
            }
        });
        
        console.log('Ended generation successfully');
    }
    
    // Update the main trace
    trace.update({
        output: {
            success: true,
            message: 'Test completed successfully'
        }
    });
    
    console.log('Updated trace successfully');
    
    // Flush traces
    console.log('Flushing traces...');
    await flushLangfuse();
    console.log('Traces flushed successfully');
    
    console.log('Langfuse integration test completed!');
    console.log('Check your Langfuse dashboard to see the test traces.');
}

// Run the test
testLangfuseIntegration().catch(console.error);
