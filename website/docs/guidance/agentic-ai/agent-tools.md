---
sidebar_label: "Agent Tools: Browser and Code Interpreter"
---

# Agent Tools: Browser and Code Interpreter

This guide continues from [Building and Deploying Agents](./building-agents.md), where we containerized and deployed an agent into our Kubernetes environment. So far our agent can only respond based on its training data. By giving it tools, we can let it browse the web and execute code — extending what it can do beyond text generation. The Agents on EKS environment includes [Selenium Grid](https://www.selenium.dev/documentation/grid/) and [Jupyter Enterprise Gateway](https://jupyter-enterprise-gateway.readthedocs.io/) for exactly this purpose.

## Prerequisites

Before starting, you should have:

- The [Agents on EKS infrastructure](https://awslabs.github.io/ai-on-eks/docs/infra/agents-on-eks) deployed with `enable_selenium_grid` and `enable_jupyter_enterprise_gateway` set to `true`
- An agent deployed and running in the cluster (see [Building and Deploying Agents](./building-agents.md))
- `kubectl` configured to access your EKS cluster

## How Strands Tools Work

The [Strands Agents SDK](https://strandsagents.com) lets you define tools as Python functions decorated with `@tool`. The agent decides when to call them based on the user's request and the tool's docstring. We'll create two tools — one for browsing the web, one for running code — and wire them into our agent.

## Add a Browser Tool

The environment runs Selenium Grid with 3 Chrome browser nodes. Selenium Grid acts as a hub that distributes browser sessions across nodes, so multiple agents can browse concurrently. Each session has a 30-minute timeout and gets its own dedicated Chrome instance.

### Connect to Selenium Grid Locally

Start by port-forwarding the Selenium Grid hub so we can develop and test the tool on our local machine:

```bash
kubectl port-forward -n selenium-grid svc/selenium-hub 4444
```

You can verify the grid is healthy by visiting `http://localhost:4444/ui` in your browser — it shows the available Chrome nodes and any active sessions.

### Update requirements.txt

Add the `selenium` package:

```text
strands-agents>=1.0.0
fastapi==0.116.1
uvicorn==0.35.0
selenium>=4.27.0
```

```bash
pip install -r requirements.txt
```

### Create browser_tools.py

We'll use an environment variable for the Selenium hub URL so the same code works both locally (via port-forward) and inside the cluster (via Kubernetes DNS):

```python
import os
from strands import tool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SELENIUM_HUB_URL = os.environ.get(
    "SELENIUM_HUB_URL",
    "http://selenium-hub.selenium-grid.svc.cluster.local:4444"
)


def _get_driver():
    options = Options()
    return webdriver.Remote(command_executor=SELENIUM_HUB_URL, options=options)


@tool
def browse_web(url: str) -> str:
    """Browse a web page and return its text content.

    Args:
        url: The URL to visit.

    Returns:
        The visible text content of the page.
    """
    driver = _get_driver()
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        return driver.find_element(By.TAG_NAME, "body").text
    finally:
        driver.quit()
```

:::tip
Always call `driver.quit()` in a `finally` block to release the browser session back to the grid. With only 3 Chrome nodes available, leaked sessions will block other agents from getting a browser.
:::

### Update agent.py

Import the browser tool and add it to the agent. We'll also update the system prompt so the agent knows it can browse:

```python
from strands import Agent
from browser_tools import browse_web


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            system_prompt=(
                "You are a helpful assistant. "
                "Use the browse_web tool to look up information from the web when needed to answer the user's request."
            ),
            tools=[browse_web]
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]
```

:::note
We've changed the system prompt from the translation-only version to a more general assistant that can use tools. The agent will decide on its own when to browse — if the user asks a question it can answer from training data, it won't use the tool. If the user asks about something current or specific, it will browse.
:::

### Test Locally

With the port-forward still running, set the environment variable and start the agent:

```bash
export SELENIUM_HUB_URL=http://localhost:4444
python entrypoint.py
```

In another terminal:

```bash
curl --location 'http://localhost:8000/' \
--header 'Content-Type: application/json' \
--data '{"request": "What is on the front page of news.ycombinator.com right now?"}'
```

The agent should call `browse_web` to visit Hacker News and return a summary of what it finds. You can watch the Selenium Grid UI at `http://localhost:4444/ui` to see the Chrome session spin up and complete.

## Add a Code Interpreter Tool

The environment runs [Jupyter Enterprise Gateway](https://jupyter-enterprise-gateway.readthedocs.io/) for remote kernel management. It provides the same execution environment as a Jupyter notebook, but driven programmatically — agents can start a Python kernel, send code to it, and get the output back.

### Connect to Enterprise Gateway Locally

Port-forward the Enterprise Gateway service:

```bash
kubectl port-forward -n enterprise-gateway svc/enterprise-gateway 8888
```

Verify it's running:

```bash
curl http://localhost:8888/api/kernelspecs
```

This should return a JSON object listing available kernel specifications (e.g., `python3`).

### Update requirements.txt

Add `websocket-client` for communicating with kernels:

```text
strands-agents>=1.0.0
fastapi==0.116.1
uvicorn==0.35.0
selenium>=4.27.0
requests>=2.32.0
websocket-client>=1.8.0
```

```bash
pip install -r requirements.txt
```

### Create code_tools.py

Like the browser tool, we use an environment variable for the gateway URL:

```python
import os
import json
import uuid
import requests
import websocket
from strands import tool

GATEWAY_URL = os.environ.get(
    "GATEWAY_URL",
    "http://enterprise-gateway.enterprise-gateway.svc.cluster.local:8888"
)


def _execute_on_kernel(code: str) -> str:
    """Start a kernel, execute code, collect output, and shut down the kernel."""
    # Start a kernel
    response = requests.post(f"{GATEWAY_URL}/api/kernels", json={"name": "python3"})
    kernel_id = response.json()["id"]

    try:
        # Connect to the kernel's WebSocket channel
        ws_url = GATEWAY_URL.replace("http://", "ws://").replace("https://", "wss://")
        ws = websocket.create_connection(f"{ws_url}/api/kernels/{kernel_id}/channels")

        # Send an execute request using the Jupyter messaging protocol
        msg_id = str(uuid.uuid4())
        ws.send(json.dumps({
            "header": {
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "agent",
                "session": str(uuid.uuid4()),
                "version": "5.3"
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True
            },
            "buffers": [],
            "channel": "shell"
        }))

        # Collect output messages until execution completes
        outputs = []
        while True:
            msg = json.loads(ws.recv())
            msg_type = msg.get("msg_type", msg.get("header", {}).get("msg_type", ""))

            if msg_type == "stream":
                outputs.append(msg["content"]["text"])
            elif msg_type == "execute_result":
                outputs.append(msg["content"]["data"].get("text/plain", ""))
            elif msg_type == "error":
                outputs.append("\n".join(msg["content"]["traceback"]))
            elif msg_type == "execute_reply":
                break

        ws.close()
        return "\n".join(outputs) if outputs else "(no output)"
    finally:
        requests.delete(f"{GATEWAY_URL}/api/kernels/{kernel_id}")


@tool
def run_python(code: str) -> str:
    """Execute Python code and return the output.

    Use this for calculations, data analysis, or any task that benefits from running code.
    Each invocation runs in a fresh kernel — define all variables and imports in the same code block.

    Args:
        code: Python code to execute.

    Returns:
        The execution output (stdout, return values) or error traceback.
    """
    return _execute_on_kernel(code)
```

Each tool invocation creates a fresh kernel and shuts it down when done. This keeps things simple and avoids state management. If you later need stateful sessions where variables persist across calls, you can hold onto the kernel ID and WebSocket connection between invocations instead of creating them each time.

### Update agent.py

Add the code tool alongside the browser tool:

```python
from strands import Agent
from browser_tools import browse_web
from code_tools import run_python


class MyAgent:
    def __init__(self):
        self.agent = Agent(
            system_prompt=(
                "You are a helpful assistant. "
                "Use the browse_web tool to look up information from the web when needed. "
                "Use the run_python tool to execute Python code for calculations, data analysis, "
                "or any task that benefits from running code. "
                "Always show your work by writing and executing code rather than doing mental math."
            ),
            tools=[browse_web, run_python]
        )

    def invoke(self, request: str) -> str:
        result = self.agent(request)
        return result.message["content"][0]["text"]
```

### Test Locally

With both port-forwards running (Selenium Grid on 4444 and Enterprise Gateway on 8888), restart the agent:

```bash
export SELENIUM_HUB_URL=http://localhost:4444
export GATEWAY_URL=http://localhost:8888
python entrypoint.py
```

Test the code interpreter:

```bash
curl --location 'http://localhost:8000/' \
--header 'Content-Type: application/json' \
--data '{"request": "What is the 50th prime number?"}'
```

The agent should use `run_python` to write and execute code that finds the answer rather than guessing. You can also test a request that combines both tools:

```bash
curl --location 'http://localhost:8000/' \
--header 'Content-Type: application/json' \
--data '{"request": "How many items are on the front page of news.ycombinator.com right now?"}'
```

The agent may browse the page with `browse_web`, then use `run_python` to count and summarize the results.

### Write a Test

Following the same testing pattern from [Best Practices](./best-practices.md), add a test for the tools. Since tools hit external services, we test them in the same way we test the agent — against the real services via port-forward:

```python
import unittest
from agent import MyAgent


class TestMyAgentWithTools(unittest.TestCase):
    def test_code_execution(self):
        agent = MyAgent()
        result = agent.invoke("What is 2 + 2? Use the code interpreter.")
        self.assertIn("4", result)

    def test_web_browsing(self):
        agent = MyAgent()
        result = agent.invoke("What is the title of the page at https://example.com?")
        self.assertIn("Example Domain", result)


if __name__ == '__main__':
    unittest.main()
```

:::note
These tests require the port-forwards to be running. In a CI/CD pipeline, the agent will be running inside the cluster where it can reach the services directly — no port-forward needed.
:::

## Deploy the Updated Agent

The deployment process follows the same pattern from [Building and Deploying Agents](./building-agents.md). Push the updated files to GitLab, let the pipeline build a new image, and update the deployment.

### Push to GitLab

```bash
git add requirements.txt agent.py browser_tools.py code_tools.py test_agent.py
git commit -m "add browser and code interpreter tools"
git push origin main
```

The CI/CD pipeline will build a new container image tagged with the commit SHA. Once the build completes, grab the new image reference from the pipeline output.

### Update the Deployment

The tools use environment variables for their service URLs, with defaults that resolve inside the cluster. Since we set the defaults to the cluster DNS names, no additional secrets or ConfigMaps are needed — the tools will work as soon as the agent pod starts.

Update the image in `deployment.yaml` and apply:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: strands-agent
  namespace: strands-agent
  labels:
    app: strands-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: strands-agent
  template:
    metadata:
      labels:
        app: strands-agent
    spec:
      containers:
        - name: agent
          image: registry.<your-domain>/root/strands-agent:<new-commit-sha>
          ports:
            - containerPort: 8000
      serviceAccountName: strands-agent
---
apiVersion: v1
kind: Service
metadata:
  name: strands-agent
  namespace: strands-agent
spec:
  ports:
    - port: 8000
      targetPort: 8000
      protocol: TCP
      name: http
  selector:
    app: strands-agent
```

```bash
kubectl apply -f deployment.yaml
```

Or update the image in place:

```bash
kubectl set image deployment/strands-agent \
  -n strands-agent \
  agent=registry.<your-domain>/root/strands-agent:<new-commit-sha>
```

:::info
Both Selenium Grid and Enterprise Gateway are internal cluster services with no ingress. Agent pods reach them directly via Kubernetes DNS — no additional network configuration or secrets are required.
:::

### Verify

Check that the pod is running with the new image:

```bash
kubectl get pods -n strands-agent
```

Port-forward the agent and test:

```bash
kubectl port-forward -n strands-agent svc/strands-agent 8000
```

```bash
curl --location 'http://localhost:8000/' \
--header 'Content-Type: application/json' \
--data '{"request": "Use the code interpreter to calculate the factorial of 20"}'
```

The agent is now running in the cluster with access to both the browser and the code interpreter, reachable by any other service in the cluster.

## Summary

Starting from the agent we built and deployed in the previous guides, we've added two tools:

1. **Browser tool** — connects to Selenium Grid for web browsing, using the `selenium` package and the `@tool` decorator
2. **Code interpreter tool** — connects to Jupyter Enterprise Gateway for Python execution, using the Jupyter messaging protocol over WebSocket

The pattern for adding tools is the same as the rest of the development workflow: develop and test locally using port-forwarding, then push to GitLab and deploy. Environment variables let the same code work in both contexts without changes.

## What's Next?

- [Tracing and Evaluating Agents](./evaluating-agents.md) — Add observability to your tool-using agent and monitor how it uses its tools
- [Best Practices for Agent Development](./best-practices.md) — Patterns for structuring agent code for testability and production readiness
