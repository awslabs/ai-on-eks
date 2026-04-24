---
sidebar_label: Best Practices for Agent Development
---

# Best Practices for Agent Development

This guide covers patterns and best practices for developing agents that transition smoothly from local development to
production on the [Agents on EKS environment](../../infra/agents-on-eks.md). We'll walk through a practical example using
the [Strands Agents SDK](https://strandsagents.com) to illustrate each pattern.

## Why AgentOps?

Building and operating AI agents at scale requires more than just inference infrastructure. Traditional DevOps practices
need to be adapted for the stochastic nature of agent outputs. Key challenges include:

- **Stochastic testing**: `expected == actual` testing won't work for LLM outputs. Evaluations require flexibility.
- **Dataset curation**: Single-shot vs multi-turn evaluations require different approaches.
- **Composite metrics**: Accuracy, tool choice, response quality, latency, and token utilization all factor into whether
  an agent is performing well.
- **Continuous evaluations**: Deployed agents need ongoing monitoring, not just pre-deployment testing.
- **Sandbox deployment**: New agent versions should be tested in isolation before replacing live agents.

The AgentOps pipeline addresses these challenges with this flow:

**Source Control → Build Image → Deploy Test → Test Dataset → Check Metrics → Redeploy Agent → Remove Test**

This ensures that every code change is built, deployed to a sandbox, evaluated against a dataset, and only promoted to
production if it meets your quality thresholds.

## Setting Up Your Development Environment

:::note
We recommend using a virtual python environment for each agent you develop. As they will eventually be built in isolated
environments, using a virtual environment helps ensure that the agent you build on your machine will work similarly when
it's deployed in a production environment. `venv` is included with python versions 3.3+ and is what will be used in this
document, but feel free to use conda or any other tool with which you're comfortable.
:::

```bash
mkdir -p ~/code/strands-agent && cd ~/code/strands-agent
python -m venv .venv
source .venv/bin/activate
```

## Pattern: Start with requirements.txt

A `requirements.txt` file is used to help install python dependencies. We recommend starting by using this file rather
than installing libraries ad-hoc using pip, as these tend to get out of sync when moving to a different python
environment or building for production.

```text
strands-agents>=1.0.0
```

```bash
pip install -r requirements.txt
```

This file becomes the single source of truth for your agent's dependencies and is used directly by the Dockerfile when
building for production.

## Example: A Simple Translation Agent

To illustrate the patterns, let's build a simple translation agent using
the [Strands Agents SDK](https://strandsagents.com) with [Amazon Bedrock](https://aws.amazon.com/bedrock/):

```python
from strands import Agent


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            model="us.amazon.nova-lite-v1:0",
            system_prompt="You are an expert translator. Translate the user request from English into Italian. Do not respond with anything other than the translation"
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]

```

While not a groundbreaking agent, this is good enough to highlight the patterns that make an agent work both locally and
in production without having to rearchitect.

## Pattern: Separate Invoke Logic from Agent

Separating the invoke logic from the agent allows you to create more complex logic, as well as give you an easy
entrypoint for testing and usage.

`agent.py`:

```python
from strands import Agent


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            model="us.amazon.nova-lite-v1:0",
            system_prompt="You are an expert translator. Translate the user request from English into Italian. Do not respond with anything other than the translation"
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]

```

`agent_entry.py`:

```python
from agent import MyAgent

agent = MyAgent()


def request(request: str):
    return agent.invoke(request)


print(request("Hello!"))
```

This allows you to leverage the agent for different purposes, similar to a library. You can separately write tests:

`test_agent.py`:

```python
import unittest
from agent import MyAgent


class TestMyAgent(unittest.TestCase):
    def test_translate(self):
        agent = MyAgent()
        result = agent.invoke("Hello!")
        self.assertEqual(result, "Ciao!")


if __name__ == '__main__':
    unittest.main()
```

Run with `python test_agent.py`.

This is a simple test, but it can detect drift from code changes (e.g., changing the prompt to ask for Spanish instead
of Italian):

```text
AssertionError: '¡Hola!' != 'Ciao!'
- ¡Hola!
+ Ciao!
```

## Pattern: API the Agent

The agent currently runs as a python script with a fixed input. To make it useful in production, we want it to:

- Take user input rather than hardcoded input
- Work over a REST API

Here's how to wrap the agent in a FastAPI endpoint:

`agent.py`:

```python
from strands import Agent


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            model="us.amazon.nova-lite-v1:0",
            system_prompt="You are an expert translator. Translate the user request from English into Italian. Do not respond with anything other than the translation"
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]

```

`agent_entry.py` (generic api wrapper)

```python
from agent import MyAgent
from api_manager import endpoint
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import json
import asyncio

agent = MyAgent()


class Request(BaseModel):
    request: str


@endpoint(method="post")
async def request(request: Request):
    return agent.invoke(request.request)

```

`entrypoint.py`:

```python
import uvicorn
from agent import app

uvicorn.run(app, host="0.0.0.0", port=8000)

```

Update `requirements.txt`:

```text
strands-agents>=1.0.0
fastapi==0.116.1
uvicorn==0.35.0
```

:::note
We separate the `api_manager.py`, `agent_entry.py` and `entrypoint.py` from the `agent.py` to make things reusable. The
first 3 files are generic and allow you to scaffold future agents, leaving `agent.py` as a script that can be used to
encapsulate any of your logic and extended from there. Feel free to structure your code however you would like. We will
continue building upon this pattern in the containerization section.
:::

Test it:

```bash
python entrypoint.py
```

```bash
curl --location 'http://localhost:8000/' \
--header 'Content-Type: application/json' \
--data '{"request": "Hello"}'
```

```text
"Ciao"
```

## Summary

At this point, you have an agent that:

1. Uses a `requirements.txt` and virtual environment for consistent, clean environments
2. Has its invoke logic separated from the main script for testability
3. Has simple tests to detect drift
4. Is wrapped in a REST API for production use

This agent is ready to be [containerized and deployed](./building-agents.md) into the Agents on EKS environment.

## What's Next?

- [Building and Deploying Agents](./building-agents.md) — Containerize, push to GitLab, and deploy to Kubernetes
