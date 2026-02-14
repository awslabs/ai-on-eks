import OpenAI from 'openai';
import 'dotenv/config';

// Create a simple test script to debug the API connection
async function testConnection() {
  console.log('Testing OpenAI API connection...');
  console.log('Base URL:', process.env.OPENAI_BASE_URL);
  console.log('API Key:', process.env.OPENAI_API_KEY ? 'Set (masked)' : 'Not set');
  
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
  });

  try {
    // First try to list models
    console.log('\nAttempting to list models...');
    const models = await openai.models.list();
    console.log('Models available:', models.data.map(m => m.id));
  } catch (error) {
    console.error('Error listing models:', error);
  }

  try {
    // Try a simple chat completion without streaming
    console.log('\nAttempting a simple chat completion...');
    const completion = await openai.chat.completions.create({
      model: 'Qwen/QwQ-32B-AWQ',
      messages: [{ role: 'user', content: 'Hello, how are you?' }],
      stream: false,
    });
    console.log('Chat completion successful:', completion);
  } catch (error) {
    console.error('Error with chat completion:', error);
    
    // Print more detailed error information
    if (error.response) {
      console.log('Response status:', error.response.status);
      console.log('Response headers:', error.response.headers);
      console.log('Response data:', error.response.data);
    }
  }
}

testConnection().catch(console.error);
