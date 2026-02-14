// Simple test script to check LLM endpoint connectivity
import 'dotenv/config';
import { OpenAI } from 'openai';

async function testEndpoint() {
  console.log('Testing LLM endpoint with a simple request');
  console.log('Base URL:', process.env.OPENAI_BASE_URL);
  
  // Create OpenAI client with simplified configuration
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_BASE_URL,
    defaultHeaders: {
      // Using only one authentication method
      "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
    }
  });

  // Simple message without tools
  const messages = [
    { role: "user", content: "Hello, can you respond with a simple greeting?" }
  ];

  try {
    console.log('Sending request...');
    
    // Try with the original model name format
    const modelName = 'Qwen/QwQ-32B-AWQ';
    console.log('Using model:', modelName);
    
    const response = await openai.chat.completions.create({
      model: modelName,
      messages: messages,
      stream: false
    });
    
    console.log('Success! Response:');
    console.log(JSON.stringify(response, null, 2));
    return response;
  } catch (error) {
    console.error('Error occurred:');
    console.error('Status:', error.status);
    console.error('Headers:', error.headers);
    
    if (error.response) {
      try {
        const responseText = await error.response.text();
        console.error('Response body:', responseText);
        try {
          const responseJson = JSON.parse(responseText);
          console.error('Response JSON:', JSON.stringify(responseJson, null, 2));
        } catch (parseError) {
          // If not valid JSON, the text version is already logged
        }
      } catch (e) {
        console.error('Could not read response data:', e);
      }
    }
    
    throw error;
  }
}

// Run the test
testEndpoint()
  .then(() => console.log('Test completed successfully'))
  .catch(err => console.error('Test failed:', err.message));
