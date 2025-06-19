# Weather MCP Server

A Model Context Protocol (MCP) server that provides weather forecasting capabilities for Amazon Q and other MCP-compatible clients.

## Features

- **Weather Forecasts**: Get detailed weather forecasts for any location
- **Weather Alerts**: Retrieve weather alerts for US states
- **Automatic Geocoding**: Automatically converts location names to coordinates
- **MCP Integration**: Seamlessly integrates with Amazon Q and other MCP clients

## Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- Amazon Q CLI (for MCP integration)

## Setup Instructions

### 1. Install uv (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### 2. Create and Activate Virtual Environment

```bash
# Create a new virtual environment with uv
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the weather MCP server and its dependencies
uv sync
```


## Amazon Q Integration

### 1. Configure MCP in Amazon Q

Create or update your Amazon Q MCP configuration file at `~/.config/amazonq/mcp.json`:

```json
{
  "mcpServers": {
    "weather": {
      "command": "uvx",
      "args": ["--no-cache", "--from", ".", "--directory", ".", "weather-mcp"]
    }
  }
}
```


### 3. Using Weather Tools in Amazon Q

Once connected, you can use the following weather-related commands in Amazon Q:

#### Get Weather Forecast
```
Get the weather forecast for Seattle, Washington
```

#### Get Weather Alerts
```
Are there any weather alerts for California?
```

#### Multiple Location Forecasts
```
Compare the weather between Miami, Florida and Denver, Colorado
```

## Available MCP Tools

The weather MCP server provides these tools to Amazon Q:

| Tool | Description | Parameters |
|------|-------------|------------|
| `weather___get_forecast` | Get weather forecast for a location | `location` (city, address, etc.) |
| `weather___get_alerts` | Get weather alerts for US states | `state` (2-letter code) |

## Example Usage


### Amazon Q Chat Examples

```
User: What's the weather like in Portland, Oregon?
Q: I'll get the current weather forecast for Portland, Oregon...

User: Are there any severe weather warnings in Florida?
Q: Let me check for weather alerts in Florida...

User: Compare the weather between Boston and Los Angeles
Q: I'll get the forecasts for both cities to compare...
```
