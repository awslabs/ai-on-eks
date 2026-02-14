import OpenAI from 'openai';
import 'dotenv/config';

// Create a test script to debug the API connection with streaming
async function testConnection() {
  console.log('Testing OpenAI API connection with streaming...');
  console.log('Base URL:', process.env.OPENAI_BASE_URL);
  console.log('API Key:', process.env.OPENAI_API_KEY ? 'Set (masked)' : 'Not set');
  
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
    defaultHeaders: {
      "api-key": process.env.OPENAI_API_KEY,
      "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
    }
  });

  try {
    // Try a simple chat completion with streaming
    console.log('\nAttempting a chat completion with streaming...');
    const stream = await openai.chat.completions.create({
      model: 'Qwen/QwQ-32B-AWQ',
      messages: [{ role: 'user', content: 'Tell me a short joke' }],
      stream: true,
    });
    
    console.log('Stream response started:');
    for await (const chunk of stream) {
      process.stdout.write(chunk.choices[0]?.delta?.content || '');
    }
    console.log('\nStream completed successfully');
  } catch (error) {
    console.error('Error with chat completion:', error);
    
    // Print more detailed error information
    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response headers:', error.response.headers);
      try {
        console.log('Response data:', await error.response.text());
      } catch (e) {
        console.log('Could not read response data');
      }
    }
  }
}

testConnection().catch(console.error);
