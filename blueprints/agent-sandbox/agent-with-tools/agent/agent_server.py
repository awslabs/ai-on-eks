"""Agent server for the agent-with-tools blueprint.

Exposes an OpenAI-compatible chat/completions API that OpenWebUI
connects to. Receives user messages, calls Amazon Bedrock Claude
with tool definitions, and dispatches tool calls to sandbox-backed
executors (code execution, Jupyter).

The server is a standard Flask app running on port 8000. OpenWebUI
is configured to use it as a custom OpenAI-compatible backend.

Environment variables:
  BEDROCK_MODEL_ID  — Claude model to use (default: Claude Sonnet 4)
  AWS_REGION        — AWS region for Bedrock (default: us-east-1)
  SANDBOX_NAMESPACE — namespace for sandbox pods (default: agent-sandboxes)
  SANDBOX_TIER      — runtime tier: runc or gvisor (default: runc)
  AGENT_PORT        — port to listen on (default: 8000)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from typing import Any

# Add user site-packages for pip install --user under readOnlyRootFilesystem
import site
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

from flask import Flask, Response, jsonify, request  # noqa: E402

from tools import execute_tool, get_tool_schemas  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AGENT_PORT = int(os.environ.get("AGENT_PORT", "8000"))

# Lazy-loaded Bedrock client
_bedrock_client = None


def get_bedrock_client():
    """Lazy-initialize the Bedrock client."""
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def call_bedrock(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call Bedrock Claude with messages and optional tool definitions.

    Translates between OpenAI message format and Bedrock's Anthropic format.
    Returns the assistant response in OpenAI format.
    """
    client = get_bedrock_client()

    # Convert OpenAI messages to Anthropic format
    system_prompt = ""
    anthropic_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            system_prompt = msg.get("content", "")
            continue

        if role == "tool":
            # Tool results go as user messages with tool_result content
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }],
            })
            continue

        if role == "assistant" and msg.get("tool_calls"):
            # Assistant message with tool calls
            content = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for tc in msg["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"],
                })
            anthropic_messages.append({"role": "assistant", "content": content})
            continue

        # Regular user or assistant message
        anthropic_messages.append({
            "role": role,
            "content": msg.get("content", ""),
        })

    # Build Bedrock request body
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": anthropic_messages,
    }
    if system_prompt:
        body["system"] = system_prompt
    if tools:
        # Convert OpenAI tool format to Anthropic format
        anthropic_tools = []
        for tool in tools:
            fn = tool["function"]
            anthropic_tools.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        body["tools"] = anthropic_tools

    resp = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    payload = json.loads(resp["body"].read())

    # Convert Anthropic response to OpenAI format
    return _anthropic_to_openai_response(payload)


def _anthropic_to_openai_response(payload: dict) -> dict:
    """Convert an Anthropic response payload to OpenAI chat completion format."""
    content_blocks = payload.get("content", [])
    text_parts = []
    tool_calls = []

    for block in content_blocks:
        if block["type"] == "text":
            text_parts.append(block["text"])
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]),
                },
            })

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish_reason = "stop"
    if payload.get("stop_reason") == "tool_use":
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": BEDROCK_MODEL_ID,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": payload.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": payload.get("usage", {}).get("output_tokens", 0),
            "total_tokens": (
                payload.get("usage", {}).get("input_tokens", 0)
                + payload.get("usage", {}).get("output_tokens", 0)
            ),
        },
    }


def process_chat(messages: list[dict], session_id: str) -> dict:
    """Process a chat request with tool-calling loop.

    Calls Bedrock, handles tool calls iteratively until the model
    produces a final text response (or hits max iterations).
    """
    tools = get_tool_schemas()
    working_messages = list(messages)
    max_iterations = 10

    for _iteration in range(max_iterations):
        response = call_bedrock(working_messages, tools=tools)
        choice = response["choices"][0]
        assistant_msg = choice["message"]

        if not assistant_msg.get("tool_calls"):
            # No tool calls — final response
            return response

        # Process tool calls
        logger.info(
            "Tool calls requested: %s",
            [tc["function"]["name"] for tc in assistant_msg["tool_calls"]],
        )

        # Add assistant message (with tool_calls) to context
        working_messages.append(assistant_msg)

        # Execute each tool and add results
        for tool_call in assistant_msg["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            fn_args = json.loads(tool_call["function"]["arguments"])

            logger.info("Executing tool: %s", fn_name)
            result = execute_tool(fn_name, fn_args, session_id)
            logger.info("Tool result (%d chars): %s...", len(result), result[:200])

            working_messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })

    # Max iterations reached
    return response


# ---------------------------------------------------------------------------
# OpenAI-compatible API endpoints
# ---------------------------------------------------------------------------

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    data = request.get_json()
    messages = data.get("messages", [])
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Add system prompt if not present
    if not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {
            "role": "system",
            "content": (
                "You are a helpful AI assistant with access to code execution tools. "
                "When the user asks you to run code, perform calculations, analyze data, "
                "or do anything that requires computation, use the available tools. "
                "For simple data analysis tasks that build on previous results, prefer "
                "the jupyter_execute tool. For one-off computations or scripts, use "
                "code_execute. Always show the user the results of tool execution."
            ),
        })

    try:
        response = process_chat(messages, session_id)
        return jsonify(response)
    except Exception as e:
        logger.exception("Chat processing failed")
        return jsonify({
            "error": {
                "message": f"Internal error: {type(e).__name__}: {e}",
                "type": "server_error",
            }
        }), 500


@app.route("/v1/models", methods=["GET"])
def list_models():
    """OpenAI-compatible models endpoint."""
    return jsonify({
        "object": "list",
        "data": [{
            "id": BEDROCK_MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "amazon-bedrock",
        }],
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    logger.info("Starting agent server on port %d", AGENT_PORT)
    logger.info("Bedrock model: %s (region: %s)", BEDROCK_MODEL_ID, AWS_REGION)
    logger.info("Tools registered: %s", list(get_tool_schemas()))
    app.run(host="0.0.0.0", port=AGENT_PORT)
