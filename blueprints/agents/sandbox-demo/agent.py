"""Reference Bedrock agent for the agent-sandbox-on-EKS showcase.

Runs inside a gVisor Sandbox pod. Exercises four paths during the
demo so the isolation + egress story is visible end-to-end:

  1. Bedrock call (allowed by ciliumnetworkpolicy-sandbox-llm.yaml) —
     the model generates a small Python snippet.
  2. Snippet execution inside the sandbox (exercises gVisor syscall
     interception via the `open` / `read` / `write` calls the snippet
     makes, which Sentry intercepts rather than routing direct to
     the host kernel).
  3. Allowed egress follow-up — the agent pip-installs `requests`
     from PyPI as a second allowed-domain call.
  4. Blocked egress attempt — the agent curls
     `demo-blocked.example.com`, which is NOT on the allowlist.
     Hubble shows this as a DROP flow, the narrator highlights it
     as the "here's the boundary holding" moment.

Each path's result is printed to stdout with a clear prefix
(``PASS:``, ``BLOCKED:``, ``ERROR:``) so the demo log-tail is
legible even to viewers who can't see the terminal clearly.

Environment variables:
  BEDROCK_MODEL_ID  — defaults to Claude Sonnet 4 in us-east-1
  AWS_REGION        — defaults to us-east-1

Expected to be invoked via ``kubectl exec`` against the sandbox
pod, not as the pod's entrypoint. This matches the SIG-Apps
singleton-stateful pattern (sandbox stays alive, agent runs as
ad-hoc interactive workload inside).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")


def step(label: str) -> None:
    """Prints a section header so the demo log-tail has clear breaks."""
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)


def call_bedrock(prompt: str) -> str | None:
    """Call Bedrock Claude Sonnet via the AWS SDK.

    Returns the assistant's text reply on success, None on failure
    (typically credential or network issue — the demo narrator
    points to Hubble at that moment to show whether the call even
    left the pod).
    """
    try:
        # pip install --user writes to $HOME/.local/lib/pythonX.Y/
        # site-packages. Python doesn't auto-include that path when
        # $HOME isn't /root, so add it explicitly before the import.
        import site  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415
        user_site = site.getusersitepackages()
        if user_site not in _sys.path:
            _sys.path.insert(0, user_site)
        import boto3  # noqa: PLC0415 — lazy so the pip-install step can
                     #                   succeed before boto3 is imported
    except ImportError:
        print("BLOCKED: boto3 not yet installed — demo ordering bug; "
              "install before calling Bedrock")
        return None

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    try:
        resp = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(resp["body"].read())
        return payload["content"][0]["text"]
    except Exception as e:  # noqa: BLE001 — demo resilience over exception hygiene
        print(f"ERROR: Bedrock call failed: {e}")
        return None


def try_egress(url: str, label: str) -> None:
    """Attempt an HTTPS GET against the URL. Prints PASS or BLOCKED
    based on the outcome. Uses a short timeout so blocked calls
    don't stall the demo waiting for the ciliumnetworkpolicy drop
    to take effect (DROP with TCP RST is near-instant, but the
    timeout floor handles the non-RST case too)."""
    print(f"Attempting egress to {url} ({label})...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "agent-sandbox-demo/0.1"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = resp.status
            body_sample = resp.read(200).decode("utf-8", errors="replace")
            print(f"PASS: {url} returned {status} ({len(body_sample)} bytes)")
    except urllib.error.URLError as e:
        reason = str(e.reason) if hasattr(e, "reason") else str(e)
        print(f"BLOCKED: {url} rejected — {reason}")
    except Exception as e:  # noqa: BLE001
        print(f"BLOCKED: {url} rejected — {type(e).__name__}: {e}")


def pip_install(package: str) -> bool:
    """Install a package inside the sandbox. Used to demonstrate the
    PyPI allowlist — a successful install means the FQDN policy
    permitted pypi.org + files.pythonhosted.org."""
    print(f"Pip-installing {package} from PyPI...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--disable-pip-version-check",
                "--user",
                package,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            print(f"PASS: {package} installed")
            return True
        tail = "\n".join(result.stderr.strip().splitlines()[-3:])
        print(f"ERROR: pip install {package} failed:\n{tail}")
        return False
    except subprocess.TimeoutExpired:
        print(f"BLOCKED: pip install {package} timed out — egress likely denied")
        return False


def execute_snippet(code: str) -> None:
    """Write the model-generated snippet to disk and run it via a
    subprocess. Two purposes:
      1. Shows the sandbox can run code (runtime story).
      2. The snippet's syscalls go through Sentry on the gVisor tier
         — useful narration point for "this is what gVisor isolation
         actually does."
    """
    snippet_path = "/tmp/agent_snippet.py"
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"Wrote snippet to {snippet_path}. Executing...")
    try:
        result = subprocess.run(
            [sys.executable, snippet_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        print("--- snippet stdout ---")
        print(result.stdout or "(empty)")
        print("--- snippet stderr ---")
        print(result.stderr or "(empty)")
        print(f"PASS: snippet exited {result.returncode}")
    except subprocess.TimeoutExpired:
        print("ERROR: snippet execution timed out (gVisor platform=ptrace "
              "is slow under heavy syscall load — acceptable for demo)")


def main() -> int:
    step("Step 1: Install boto3 from PyPI (allowed egress)")
    if not pip_install("boto3"):
        # If PyPI is blocked, the demo is broken — fail loudly so the
        # narrator can pivot to Hubble to show WHY.
        print("\nFATAL: PyPI install failed — check CiliumNetworkPolicy "
              "sandbox-llm-allowlist in the agent-sandboxes namespace.\n"
              "Hubble should show DROP flows to pypi.org if the policy "
              "isn't permitting it.\n")
        return 1

    step("Step 2: Call Bedrock Claude Sonnet (allowed egress)")
    prompt = (
        "Write a short Python function called `count_words(text)` that "
        "returns the number of whitespace-separated words in a string. "
        "Include a simple test call at the bottom. Reply with ONLY the "
        "Python code, no explanation or markdown fences."
    )
    snippet = call_bedrock(prompt)
    if not snippet:
        print("\nFATAL: Bedrock call failed — check IAM permissions "
              "(bedrock:InvokeModel) and the CiliumNetworkPolicy for "
              "bedrock-runtime.us-east-1.amazonaws.com.\n")
        return 1
    print("--- Bedrock reply ---")
    print(snippet)

    step("Step 3: Execute model-generated snippet inside the sandbox")
    # Strip any accidental markdown fences the model added despite the
    # explicit "no markdown" instruction — real-world defense.
    cleaned = snippet.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline > 0 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()
    execute_snippet(cleaned)

    step("Step 4: Attempt egress to a BLOCKED domain")
    # This should get denied by the CiliumNetworkPolicy. Hubble shows
    # the DROP flow with reason "policy-denied". Demo money shot.
    try_egress("https://demo-blocked.example.com/", "NOT on allowlist")

    step("Demo complete")
    print("\nExpected outcomes:")
    print("  Step 1 (PyPI):            PASS — allowed by FQDN policy")
    print("  Step 2 (Bedrock):         PASS — allowed by FQDN policy")
    print("  Step 3 (snippet exec):    PASS — syscalls via Sentry (gVisor)")
    print("  Step 4 (blocked egress):  BLOCKED — denied by FQDN policy")
    print("\nCheck Hubble UI for the full flow decision trail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
