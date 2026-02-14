import OpenAI from 'openai';
import 'dotenv/config';

// Create a test script to debug the API connection with tools
async function testConnection() {
  console.log('Testing OpenAI API connection with tools...');
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

  // Define a simple tool
  const tools = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: {
              type: "string",
              enum: ["celsius", "fahrenheit"],
              description: "The temperature unit to use",
            },
          },
          required: ["location"],
        },
      },
    }
  ];

  try {
    // Try a simple chat completion without streaming
    console.log('\nAttempting a chat completion with tools...');
    const completion = await openai.chat.completions.create({
      model: 'Qwen/QwQ-32B-AWQ',
      messages: [{ role: 'user', content: 'What\'s the weather like in Seattle?' }],
      stream: false,
      tools: tools,
    });
    console.log('Chat completion successful:', JSON.stringify(completion, null, 2));
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
