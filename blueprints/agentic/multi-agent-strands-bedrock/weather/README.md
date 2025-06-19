# Weather Agent

A weather assistant built with Strands Agents, MCP (Model Context Protocol), and A2A (Agent to Agent) for providing weather forecasts and alerts.

## Features

- Weather forecasts and alerts
- Interactive Weather Agent
- MCP server for Weather Agent as MCP Tool
- A2A server exposing the Weather Agent

## Usage

# Install dependencies
```bash
uv sync
```

# Run interactive mode
```bash
uv run agent_interactive.py
```
using uvx
```bash
uvx --no-cache --from . --directory . weather-agent-interactive
```

# Run as mcp server
```bash
uv run agent_mcp_server.py
```
using uvx
```bash
uvx --no-cache --from . --directory . weather-agent-mcp-server --transport streamable-http
```

# Run as a2a server
```bash
uv run agent_a2a_server.py
```
using uvx
```bash
uvx --no-cache --from . --directory . weather-agent-a2a-server
```

# Run the a2a client
```bash
uv run test_a2a_client.py
```

# Running in a Container

Build the container using docker
```bash
docker build . --tag agent
```
Build the container using finch
```bash
finch build . --tag agent
```

Run the agent as mcp server
```bash
docker run \
-v $HOME/.aws:/app/.aws \
-p 8080:8080 \
-e AWS_REGION=${AWS_REGION} \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
-e DEBUG=1 \
agent weather-agent-mcp-server --transport streamable-http
```
Connect your mcp client such as `npx @modelcontextprotocol/inspector`

Run the agent as a2a server
```bash
docker run \
-v $HOME/.aws:/app/.aws \
-p 9000:9000 \
-e AWS_REGION=${AWS_REGION} \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
-e DEBUG=1 \
agent weather-agent-a2a-server
```
Then test in another terminal running `uv run test_a2a_client.py`

Run the agent interactive
```bash
docker run -it \
-v $HOME/.aws:/app/.aws \
-e AWS_REGION=${AWS_REGION} \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
-e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
-e AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
agent weather-agent-interactive
```
Type a question, to exit use `/quit`
