import os
import json

from mcp import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from strands import Agent, tool
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient


@tool
def weather_assistant_tool(query: str) -> str:
    """
    Process and respond Weather forecast or alerts

    Args:
        query: The user's question

    Returns:
        A helpful response addressing user query
    """
    return weather_assistant(query)


def weather_assistant(query: str) -> str:
    """ Process and respond Weather forecast or alerts """
    weather_agent = get_weather_agent()
    response = str(weather_agent(query))
    if len(response) > 0:
        return response

    return "I apologize, but I couldn't properly analyze your question. Could you please rephrase or provide more context?"


def get_weather_agent() -> Agent:
    """Get the Weather Agent"""
    #model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
    model_id = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(model_id=model_id)

    try:
        mcp_server_url = os.getenv("MCP_SERVER_URL")
        if mcp_server_url:
            from mcp.client.streamable_http import streamablehttp_client
            mcp_client = MCPClient(
                lambda: streamablehttp_client(mcp_server_url)
            )
        else:
            mcp_client = MCPClient(
                lambda: stdio_client(
                    StdioServerParameters(
                        command="uvx",
                        args=["--from",".","--directory","weather-mcp-server","weather-mcp"]
                    )
                )
            )



        mcp_client.start()


        tools = mcp_client.list_tools_sync()
        # Create the research agent with specific capabilities
        weather_agent = Agent(
            model=bedrock_model,
            system_prompt="""You are Weather Assistant helps the user with forecast or alerts:
            - weather forecast for an US City for the next 3 days if not specify which period
            - when returning forecast always include for each item if the weather is good for outdoor activities or not, this is useful information for the user to know for each day
            - know about weather alerts given an US City
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



if __name__ == "__main__":
  weather_assistant("Get the weather for Seattle")
