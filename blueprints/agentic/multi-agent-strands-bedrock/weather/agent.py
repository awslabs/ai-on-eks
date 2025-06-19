"""Weather agent module for providing weather forecasts and alerts."""

import os
from typing import Optional

from mcp import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient


@tool
def weather_assistant_tool(query: str) -> str:
    """
    Process and respond to weather forecast or alert queries.

    Args:
        query: The user's weather-related question

    Returns:
        A helpful response addressing the user's query
    """
    return weather_assistant(query)


def weather_assistant(query: str) -> str:
    """Process and respond to weather forecast or alert queries."""
    try:
        weather_agent = get_weather_agent()
        response = str(weather_agent(query))
        if response:
            return response
    except Exception as e:
        print(f"Error processing weather query: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again later."

    return "I apologize, but I couldn't properly analyze your question. Could you please rephrase or provide more context?"


def get_weather_agent() -> Agent:
    """
    Create and return a Weather Agent instance.

    Returns:
        Agent: A configured weather assistant agent
    """
    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(model_id=model_id)

    try:
        mcp_client = _create_mcp_client()
        mcp_client.start()
        tools = mcp_client.list_tools_sync()

        # Create the weather agent with specific capabilities
        weather_agent = Agent(
            model=bedrock_model,
            system_prompt="""You are Weather Assistant that helps the user with forecasts or alerts:
- Provide weather forecasts for US cities for the next 3 days if no specific period is mentioned
- When returning forecasts, always include whether the weather is good for outdoor activities for each day
- Provide information about weather alerts for US cities when requested
""",
            tools=tools,
        )

        return weather_agent

    except Exception as e:
        print(f"Error getting agent: {str(e)}")
        # Return a fallback agent when MCP client fails
        fallback_agent = Agent(
            model=bedrock_model,
            system_prompt="""I am a Weather Assistant, but I'm currently experiencing technical difficulties accessing weather data.
I apologize for the inconvenience. Please try again later or contact support if the issue persists.""",
            tools=[],
        )
        return fallback_agent


def _create_mcp_client() -> MCPClient:
    """
    Create an MCP client based on environment configuration.

    Returns:
        MCPClient: Configured MCP client
    """
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    if mcp_server_url:
        return MCPClient(
            lambda: streamablehttp_client(mcp_server_url)
        )

    mcp_server_location = os.getenv("MCP_SERVER_LOCATION","weather-mcp-server")
    return MCPClient(
        lambda: stdio_client(
            StdioServerParameters(
                command="uvx",
                args=["--from", ".", "--directory", mcp_server_location, "weather-mcp"]
            )
        )
    )


if __name__ == "__main__":
    weather_assistant("Get the weather for Seattle")
