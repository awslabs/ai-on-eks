"""MCP server implementation for the Weather Agent."""

import argparse
import os

from mcp.server.fastmcp import FastMCP

from agent import weather_assistant as agent

# Initialize FastMCP server
mcp = FastMCP("weather-agent")


@mcp.tool()
async def weather(query: str) -> str:
    """
    Process and respond to weather forecast or alert queries.

    Args:
        query: The user's input

    Returns:
        A response to the user's weather query
    """
    return agent(query)


def weather_mcp_server():
    """Main entry point for the weather MCP server."""
    print("Starting weather MCP server...")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Weather MCP Server')
    parser.add_argument(
        '--transport',
        choices=['stdio', 'streamable-http'],
        default='stdio',
        help='Transport protocol to use stdio(default) or streamable-http'
    )

    args = parser.parse_args()

    # Run MCP server with specified transport
    mcp.settings.port = int(os.getenv("MCP_PORT", "8080"))
    mcp.settings.host = '0.0.0.0'
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    weather_mcp_server()
