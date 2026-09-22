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

# Reuse a data-analysis sandbox pod per session to avoid repeated cold starts.
# Note: this reuses the pod, not interpreter state — each call runs a fresh
# `python` process, so variables/imports do not carry over between calls.
_data_analysis_sessions: dict[str, str] = {}  # session_id -> pod_name


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


def _resolve_sandbox_pod(claim_name: str, timeout: int = 60) -> str | None:
    """Resolve the sandbox pod backing a SandboxClaim.

    agent-sandbox v1beta1: the pod's name always equals the sandbox's
    name, but the sandbox's name only equals the claim's name on the
    cold-start path. Warm-pool checkouts adopt a pool-created sandbox
    with a pool-derived name — so never assume; read it from the
    claim's status.sandbox.name (populated once the claim binds).

    Returns the pod name, or None if the claim doesn't exist or didn't
    bind within the timeout.
    """
    deadline = time.monotonic() + timeout
    while True:
        result = _kubectl(
            "-n", NS, "get", f"sandboxclaim/{claim_name}",
            "-o", "jsonpath={.status.sandbox.name}",
        )
        if result.returncode != 0:
            return None  # claim doesn't exist
        name = result.stdout.strip()
        if name:
            return name
        if time.monotonic() >= deadline:
            return None
        time.sleep(2)


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
    claim_name = f"code-exec-{session_id[:8]}"

    # Resolve the sandbox pod from the claim; create the claim if needed.
    pod_name = _resolve_sandbox_pod(claim_name, timeout=5)
    if pod_name is None or not _pod_exists(pod_name):
        logger.info("Creating code-execution sandbox claim: %s", claim_name)
        claim_manifest = _render_code_exec_claim(claim_name)
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

        pod_name = _resolve_sandbox_pod(claim_name)
        if pod_name is None:
            return "ERROR: SandboxClaim did not bind a sandbox within 60s"
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


def _render_code_exec_claim(claim_name: str) -> str:
    """Render a SandboxClaim manifest for a code-execution sandbox.

    v1beta1: claims check out from a SandboxWarmPool (the pool ships
    alongside the SandboxTemplate in manifests/ as <template>-pool).
    """
    template = SANDBOX_TEMPLATE_CODE.replace('__TIER__', os.environ.get('SANDBOX_TIER', 'runc'))
    return f"""apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxClaim
metadata:
  name: {claim_name}
  namespace: {NS}
  labels:
    agent-sandbox/role: code-exec
    agent-sandbox/managed-by: agent-with-tools
spec:
  warmPoolRef:
    name: {template}-pool
"""


# ---------------------------------------------------------------------------
# Data Analysis Execution Tool
# ---------------------------------------------------------------------------

DATA_ANALYSIS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "data_analysis_execute",
        "description": (
            "Execute Python code in a data-analysis sandbox with common data "
            "science libraries available (numpy, pandas, matplotlib). Use this "
            "for data analysis, computation, and plotting. Each call runs "
            "independently — variables and imports do NOT carry over between "
            "calls, so include everything a snippet needs (imports, data setup) "
            "in the same call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to execute.",
                },
            },
            "required": ["code"],
        },
    },
}


def execute_data_analysis(args: dict[str, Any], session_id: str) -> str:
    """Execute Python code in a data-analysis sandbox.

    The sandbox pod is reused across calls within the same session_id
    (avoiding repeated cold starts), but each call runs a *fresh* Python
    interpreter — the code is written to a file and run with `python`, so
    interpreter state (variables, imports, dataframes) does NOT persist
    between calls. Each snippet must be self-contained.
    """
    code = args.get("code", "")
    pod_name = _data_analysis_sessions.get(session_id)

    if pod_name is None or not _pod_exists(pod_name):
        claim_name = f"data-analysis-{session_id[:8]}"
        logger.info("Creating data-analysis sandbox claim: %s", claim_name)

        claim_manifest = _render_data_analysis_claim(claim_name)
        result = subprocess.run(
            ["kubectl", "apply", "-f", "-"],
            input=claim_manifest,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return f"ERROR: Failed to create data-analysis sandbox: {result.stderr}"

        pod_name = _resolve_sandbox_pod(claim_name)
        if pod_name is None:
            return "ERROR: data-analysis SandboxClaim did not bind a sandbox within 60s"
        if not _wait_for_pod(pod_name, timeout=240):
            return "ERROR: data-analysis sandbox pod did not become Ready within 4 minutes"

        _data_analysis_sessions[session_id] = pod_name

    # Write the snippet to a file and run it with a fresh interpreter.
    write_result = _kubectl(
        "-n", NS, "exec", pod_name, "-c", "data-analysis-runtime", "--",
        "/bin/sh", "-c",
        f"cat > /tmp/analysis_cell.py << 'AGENT_ANALYSIS_EOF'\n{code}\nAGENT_ANALYSIS_EOF",
        timeout=10,
    )
    if write_result.returncode != 0:
        return f"ERROR: Failed to write code to data-analysis sandbox: {write_result.stderr}"

    result = _kubectl(
        "-n", NS, "exec", pod_name, "-c", "data-analysis-runtime", "--",
        "python", "/tmp/analysis_cell.py",
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


def _render_data_analysis_claim(claim_name: str) -> str:
    """Render a SandboxClaim manifest for a data-analysis sandbox.

    v1beta1: claims check out from a SandboxWarmPool (the pool ships
    alongside the SandboxTemplate in manifests/ as <template>-pool).
    """
    template = SANDBOX_TEMPLATE_JUPYTER.replace('__TIER__', os.environ.get('SANDBOX_TIER', 'runc'))
    return f"""apiVersion: extensions.agents.x-k8s.io/v1beta1
kind: SandboxClaim
metadata:
  name: {claim_name}
  namespace: {NS}
  labels:
    agent-sandbox/role: data-analysis
    agent-sandbox/managed-by: agent-with-tools
spec:
  warmPoolRef:
    name: {template}-pool
"""


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

TOOLS = {
    "code_execute": {
        "schema": CODE_EXEC_SCHEMA,
        "execute": execute_code,
    },
    "data_analysis_execute": {
        "schema": DATA_ANALYSIS_SCHEMA,
        "execute": execute_data_analysis,
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
