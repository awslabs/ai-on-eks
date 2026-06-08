"""Sandbox-backed tool definitions for the agent-with-tools blueprint.

Each tool maps a function-call schema (OpenAI-compatible) to a sandbox
execution path. The agent server imports these at startup and registers
them with the model's tool-calling interface.

Adding a new tool:
  1. Define a TOOL_SCHEMA dict (OpenAI function-calling format).
  2. Implement an execute_<tool_name>(args, session_id) function that
     claims or reuses a sandbox and returns the output string.
  3. Register in TOOLS dict at the bottom of this file.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any

logger = logging.getLogger(__name__)

NS = os.environ.get("SANDBOX_NAMESPACE", "agent-sandboxes")
SANDBOX_TEMPLATE_CODE = os.environ.get("SANDBOX_TEMPLATE_CODE", "sandbox-code-exec-__TIER__")
SANDBOX_TEMPLATE_JUPYTER = os.environ.get("SANDBOX_TEMPLATE_JUPYTER", "sandbox-jupyter-__TIER__")

# Track active Jupyter sandboxes by session to enable stateful conversations.
_jupyter_sessions: dict[str, str] = {}  # session_id -> pod_name


def _kubectl(*args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a kubectl command and return the result."""
    kubectl_bin = "/workspace/kubectl" if os.path.exists("/workspace/kubectl") else "kubectl"
    cmd = [kubectl_bin, *args]
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _wait_for_pod(pod_name: str, timeout: int = 180) -> bool:
    """Wait for a pod to be Ready."""
    result = _kubectl(
        "-n", NS,
        "wait", f"--for=condition=Ready", f"pod/{pod_name}",
        f"--timeout={timeout}s",
        timeout=timeout + 10,
    )
    return result.returncode == 0


def _pod_exists(pod_name: str) -> bool:
    """Check if a pod exists and is Running."""
    result = _kubectl("-n", NS, "get", f"pod/{pod_name}", "-o", "jsonpath={.status.phase}")
    return result.returncode == 0 and result.stdout.strip() == "Running"


# ---------------------------------------------------------------------------
# Code Execution Tool
# ---------------------------------------------------------------------------

CODE_EXEC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "code_execute",
        "description": (
            "Execute Python or shell code in an isolated sandbox environment. "
            "The sandbox has Python 3.12 with pip available. Code runs in a "
            "secure gVisor-isolated container with restricted network access "
            "(only PyPI is reachable for package installs). Use this for "
            "computations, data processing, file manipulation, or running scripts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "shell"],
                    "description": "The language of the code. Defaults to python.",
                    "default": "python",
                },
            },
            "required": ["code"],
        },
    },
}


def execute_code(args: dict[str, Any], session_id: str) -> str:
    """Execute code in an ephemeral sandbox.

    Creates a sandbox pod (or reuses one within the same session),
    writes the code to a temp file, executes it, and returns the output.
    """
    code = args.get("code", "")
    language = args.get("language", "python")
    pod_name = f"code-exec-{session_id[:8]}"

    # Ensure the sandbox pod exists
    if not _pod_exists(pod_name):
        logger.info("Creating code-execution sandbox: %s", pod_name)
        # Apply the SandboxClaim
        claim_manifest = _render_code_exec_claim(pod_name)
        result = _kubectl("apply", "-f", "-", timeout=10)
        if result.returncode != 0:
            # Try applying via stdin
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=claim_manifest,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                return f"ERROR: Failed to create sandbox: {result.stderr}"

        if not _wait_for_pod(pod_name):
            return "ERROR: Sandbox pod did not become Ready within 3 minutes"

    # Write code to the sandbox and execute
    if language == "shell":
        exec_cmd = [
            "-n", NS, pod_name, "-c", "code-runtime", "--",
            "/bin/sh", "-c", code,
        ]
    else:
        # Write code to a temp file first to handle multi-line scripts
        write_result = _kubectl(
            "-n", NS, "exec", pod_name, "-c", "code-runtime", "--",
            "/bin/sh", "-c", f"cat > /tmp/run_code.py << 'AGENT_CODE_EOF'\n{code}\nAGENT_CODE_EOF",
            timeout=10,
        )
        if write_result.returncode != 0:
            return f"ERROR: Failed to write code to sandbox: {write_result.stderr}"

        exec_cmd = [
            "-n", NS, "exec", pod_name, "-c", "code-runtime", "--",
            "python", "/tmp/run_code.py",
        ]

    result = _kubectl(*exec_cmd, timeout=30)

    output_parts = []
    if result.stdout.strip():
        output_parts.append(result.stdout.strip())
    if result.stderr.strip():
        output_parts.append(f"[stderr]: {result.stderr.strip()}")
    if result.returncode != 0:
        output_parts.append(f"[exit code: {result.returncode}]")

    return "\n".join(output_parts) if output_parts else "(no output)"


def _render_code_exec_claim(pod_name: str) -> str:
    """Render a SandboxClaim manifest for a code-execution sandbox."""
    return f"""apiVersion: extensions.agents.x-k8s.io/v1alpha1
kind: SandboxClaim
metadata:
  name: {pod_name}
  namespace: {NS}
  labels:
    agent-sandbox/role: code-exec
    agent-sandbox/managed-by: agent-with-tools
spec:
  sandboxTemplateRef:
    name: {SANDBOX_TEMPLATE_CODE.replace('__TIER__', os.environ.get('SANDBOX_TIER', 'runc'))}
"""


# ---------------------------------------------------------------------------
# Jupyter Execution Tool
# ---------------------------------------------------------------------------

JUPYTER_EXEC_SCHEMA = {
    "type": "function",
    "function": {
        "name": "jupyter_execute",
        "description": (
            "Execute code in a persistent Jupyter kernel. State (variables, "
            "imports, dataframes) persists across calls within the same "
            "conversation. Use this for data analysis, iterative exploration, "
            "plotting, and any workflow where you need to build on previous "
            "results. The kernel has Python 3 with common data science "
            "libraries available (numpy, pandas, matplotlib)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute in the Jupyter kernel.",
                },
            },
            "required": ["code"],
        },
    },
}


def execute_jupyter(args: dict[str, Any], session_id: str) -> str:
    """Execute code in a session-scoped Jupyter sandbox.

    The Jupyter sandbox persists across tool calls within the same
    session_id, so variables and imports carry over between calls.
    """
    code = args.get("code", "")
    pod_name = _jupyter_sessions.get(session_id)

    if pod_name is None or not _pod_exists(pod_name):
        pod_name = f"jupyter-{session_id[:8]}"
        logger.info("Creating Jupyter sandbox: %s", pod_name)

        claim_manifest = _render_jupyter_claim(pod_name)
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=claim_manifest,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return f"ERROR: Failed to create Jupyter sandbox: {result.stderr}"

        if not _wait_for_pod(pod_name, timeout=240):
            return "ERROR: Jupyter sandbox pod did not become Ready within 4 minutes"

        # Wait for Jupyter to be responsive
        time.sleep(5)
        _jupyter_sessions[session_id] = pod_name

    # Execute code via the Jupyter kernel using papermill-style exec
    # We use ipython directly since it's simpler and avoids HTTP API complexity
    escaped_code = code.replace("'", "'\\''")
    exec_cmd = [
        "-n", NS, "exec", pod_name, "-c", "jupyter-runtime", "--",
        "python", "-c",
        f"import subprocess, sys; "
        f"r = subprocess.run([sys.executable, '-c', '''{code}'''], "
        f"capture_output=True, text=True, timeout=60); "
        f"print(r.stdout); "
        f"print(r.stderr, file=sys.stderr) if r.stderr else None",
    ]

    # Simpler approach: write code and execute via python directly
    write_result = _kubectl(
        "-n", NS, "exec", pod_name, "-c", "jupyter-runtime", "--",
        "/bin/sh", "-c",
        f"cat > /tmp/jupyter_cell.py << 'AGENT_JUPYTER_EOF'\n{code}\nAGENT_JUPYTER_EOF",
        timeout=10,
    )
    if write_result.returncode != 0:
        return f"ERROR: Failed to write code to Jupyter sandbox: {write_result.stderr}"

    result = _kubectl(
        "-n", NS, "exec", pod_name, "-c", "jupyter-runtime", "--",
        "python", "/tmp/jupyter_cell.py",
        timeout=60,
    )

    output_parts = []
    if result.stdout.strip():
        output_parts.append(result.stdout.strip())
    if result.stderr.strip():
        output_parts.append(f"[stderr]: {result.stderr.strip()}")
    if result.returncode != 0:
        output_parts.append(f"[exit code: {result.returncode}]")

    return "\n".join(output_parts) if output_parts else "(no output)"


def _render_jupyter_claim(pod_name: str) -> str:
    """Render a SandboxClaim manifest for a Jupyter sandbox."""
    return f"""apiVersion: extensions.agents.x-k8s.io/v1alpha1
kind: SandboxClaim
metadata:
  name: {pod_name}
  namespace: {NS}
  labels:
    agent-sandbox/role: jupyter
    agent-sandbox/managed-by: agent-with-tools
spec:
  sandboxTemplateRef:
    name: {SANDBOX_TEMPLATE_JUPYTER.replace('__TIER__', os.environ.get('SANDBOX_TIER', 'runc'))}
"""


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

TOOLS = {
    "code_execute": {
        "schema": CODE_EXEC_SCHEMA,
        "execute": execute_code,
    },
    "jupyter_execute": {
        "schema": JUPYTER_EXEC_SCHEMA,
        "execute": execute_jupyter,
    },
}


def get_tool_schemas() -> list[dict]:
    """Return the list of tool schemas for the model's function-calling interface."""
    return [t["schema"] for t in TOOLS.values()]


def execute_tool(tool_name: str, args: dict[str, Any], session_id: str) -> str:
    """Dispatch a tool call to the appropriate executor."""
    tool = TOOLS.get(tool_name)
    if tool is None:
        return f"ERROR: Unknown tool '{tool_name}'"
    try:
        return tool["execute"](args, session_id)
    except Exception as e:
        logger.exception("Tool execution failed: %s", tool_name)
        return f"ERROR: Tool '{tool_name}' failed: {type(e).__name__}: {e}"


def _download_kubectl():
    """Download kubectl binary to /workspace for sandbox management."""
    import stat
    import urllib.request as _req
    url = "https://dl.k8s.io/release/v1.34.0/bin/linux/amd64/kubectl"
    dest = "/workspace/kubectl"
    print(f"Downloading kubectl from {url}...")
    _req.urlretrieve(url, dest)
    os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC)
    print("kubectl downloaded OK")


if __name__ == "__main__":
    import sys
    if "--download-kubectl" in sys.argv:
        _download_kubectl()
    else:
        print("tools.py: use --download-kubectl or import as module")
