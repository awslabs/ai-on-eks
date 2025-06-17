import os
from agent import weather_assistant as agent
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("weather-agent")

@mcp.tool()
async def weather(query: str) -> str:
    """
    Process and respond Weather forecast or alerts

    Args:
        query: The user's input
    """
    return agent(query)

def weather_mcp_server():
    """Main entry point for the weather MCP server."""
    print("Starting weather MCP server...")
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Weather MCP Server')
    parser.add_argument('--transport',
                       choices=['stdio','streamable-http'],
                       default='stdio',
                       help='Transport protocol to use stdio(default) or streamable-http')

    args = parser.parse_args()

    # Run MCP server with specified transport
    mcp.settings.port = int(os.getenv("MCP_PORT", "8080"))
    mcp.settings.host = '0.0.0.0'
    mcp.run(transport=args.transport)

if __name__ == "__main__":
    weather_mcp_server()
