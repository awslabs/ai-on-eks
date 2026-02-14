// Test script to check MCP server functionality
import { spawn } from 'child_process';
import path from 'path';

const outPath = path.resolve(process.cwd(), 'output');
console.log(`Testing MCP server with output path: ${outPath}`);

// Start the MCP server process
const mcpProcess = spawn('npx', ['-y', '@modelcontextprotocol/server-filesystem', outPath], {
  stdio: ['pipe', 'pipe', 'pipe']
});

// Log server output
mcpProcess.stdout.on('data', (data) => {
  console.log(`MCP Server stdout: ${data}`);
});

mcpProcess.stderr.on('data', (data) => {
  console.error(`MCP Server stderr: ${data}`);
});

// Send a test message to the server after it starts
setTimeout(() => {
  try {
    console.log('Sending test message to MCP server...');
    const message = {
      jsonrpc: '2.0',
      id: '1',
      method: 'listTools',
      params: {}
    };
    
    mcpProcess.stdin.write(JSON.stringify(message) + '\n');
    
    // Wait for response and then try to call the write_file tool
    setTimeout(() => {
      console.log('Attempting to call write_file tool...');
      const writeFileMessage = {
        jsonrpc: '2.0',
        id: '2',
        method: 'callTool',
        params: {
          name: 'write_file',
          arguments: {
            path: 'mcp-test.md',
            content: '# MCP Test\n\nThis file was created using the MCP server.'
          }
        }
      };
      
      mcpProcess.stdin.write(JSON.stringify(writeFileMessage) + '\n');
      
      // Close the server after testing
      setTimeout(() => {
        console.log('Closing MCP server...');
        mcpProcess.kill();
        process.exit(0);
      }, 2000);
    }, 1000);
  } catch (error) {
    console.error(`Error sending message to MCP server: ${error}`);
    mcpProcess.kill();
    process.exit(1);
  }
}, 2000);
