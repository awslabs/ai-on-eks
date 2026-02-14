import OpenAI from 'openai';
import 'dotenv/config';

// Create a simple test script to debug the API connection with authentication
async function testConnection() {
  console.log('Testing OpenAI API connection with authentication...');
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
