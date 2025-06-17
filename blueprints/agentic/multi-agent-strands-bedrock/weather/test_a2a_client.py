"""Test client for Agent-to-Agent (A2A) communication."""

import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)
from rich.console import Console
from rich.markdown import Markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"
BASE_URL = f"http://localhost:{os.getenv('A2A_PORT', '9000')}"


async def main() -> None:
    """Run the A2A client test."""
    # Set a longer timeout for the HTTP client
    timeout = httpx.Timeout(60.0)  # 60 seconds timeout
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=BASE_URL,
        )

        # Fetch Public Agent Card and Initialize Client
        agent_card = await fetch_agent_card(resolver)
        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        logger.info("A2AClient initialized.")

        # Prepare and send message
        request = create_message_request()
        response = await client.send_message(request)

        # Process and display response
        display_response(response)


async def fetch_agent_card(resolver: A2ACardResolver) -> AgentCard:
    """
    Fetch the agent card from the server.

    Args:
        resolver: The A2ACardResolver to use for fetching

    Returns:
        The fetched agent card

    Raises:
        RuntimeError: If fetching the agent card fails
    """
    try:
        logger.info("Attempting to fetch public agent card from: %s %s", BASE_URL, PUBLIC_AGENT_CARD_PATH)
        agent_card = await resolver.get_agent_card()  # Fetches from default public path
        logger.info("Successfully fetched public agent card:")
        logger.info(agent_card.model_dump_json(indent=2, exclude_none=True))
        return agent_card
    except Exception as e:
        logger.exception("Critical error fetching public agent card")
        raise RuntimeError("Failed to fetch the public agent card. Cannot continue.") from e


def create_message_request() -> SendMessageRequest:
    """
    Create a message request to send to the agent.

    Returns:
        A SendMessageRequest object
    """
    send_message_payload: dict[str, Any] = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": "How is the weather for the rest of the week in Las Vegas?"}],
            "messageId": uuid4().hex,
        },
    }
    return SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))


def display_response(response: Any) -> None:
    """
    Display the response from the agent.

    Args:
        response: The response from the agent
    """
    # Print the full JSON response
    response_json = response.model_dump_json(indent=2, exclude_none=True)
    print(response_json)

    # Parse the JSON response to extract the text content
    response_dict = json.loads(response_json)

    # Extract and render the markdown text
    if "result" in response_dict and "parts" in response_dict["result"]:
        for part in response_dict["result"]["parts"]:
            if part.get("kind") == "text" and "text" in part:
                print("\n\n=== RENDERED MARKDOWN ===\n")
                console = Console()
                md_text = part["text"]
                # Render markdown using rich's Markdown class
                console.print(Markdown(md_text))
                print("\n=== END OF MARKDOWN ===\n")


if __name__ == "__main__":
    asyncio.run(main())
