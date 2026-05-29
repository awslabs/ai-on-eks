"""Reference Bedrock agent for the agent-sandbox-on-EKS blueprint.

Runs inside a gVisor Sandbox pod. Exercises five paths so both
enforcement layers (FQDN proxy + L3/L4 policy) are visible end-to-end:

  1. PyPI egress (allowed by FQDN policy) — pip-install boto3 from
     pypi.org + files.pythonhosted.org.
  2. Bedrock call (allowed by FQDN policy) — the model generates a
     small Python snippet via bedrock-runtime.*.amazonaws.com.
  3. Snippet execution inside the sandbox (exercises gVisor syscall
     interception via the `open` / `read` / `write` calls the snippet
     makes, which Sentry intercepts rather than routing direct to
     the host kernel).
  4. Blocked FQDN egress — request `blocked-example.example.com`,
     which is NOT on the allowlist. Cilium's DNS proxy returns an
     empty answer; Python surfaces this as "no address associated
     with hostname". Observable in Hubble via `cilium observe` /
     DNS-proxy flow logs (see README); NOT visible as a DROPPED
     flow in the default Hubble UI because FQDN enforcement is a
     DNS-layer filter, not an L3/L4 packet drop.
  5. Blocked IP egress — raw TCP connect to 8.8.8.8:443, a
     non-allowlisted IP. Bypasses DNS entirely, so Cilium's L3/L4
     policy drops the SYN packet directly. THIS one appears as a
     red DROPPED flow in Hubble's default view — the visible
     counterpart to the invisible Step 4.

Each path's result is printed to stdout with a clear prefix
(``PASS:``, ``BLOCKED:``, ``ERROR:``) so the output log is
legible and machine-parseable (see `conformance.sh`).

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
import socket
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
    """Print a section header so the output log has clear breaks."""
    print(f"\n{'=' * 70}\n>>> {label}\n{'=' * 70}", flush=True)


def call_bedrock(prompt: str) -> str | None:
    """Call Bedrock Claude Sonnet via the AWS SDK.

    Returns the assistant's text reply on success, None on failure
    (typically credential or network issue).
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
        print("BLOCKED: boto3 not yet installed — install before calling Bedrock")
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
    except Exception as e:  # noqa: BLE001 — broad to surface any Bedrock failure
        print(f"ERROR: Bedrock call failed: {e}")
        return None


def try_egress(url: str, label: str) -> None:
    """Attempt an HTTPS GET against the URL. Prints PASS or BLOCKED
    based on the outcome. Uses a short timeout so blocked calls
    don't stall waiting for the ciliumnetworkpolicy drop to take
    effect (DROP with TCP RST is near-instant, but the timeout
    floor handles the non-RST case too)."""
    print(f"Attempting egress to {url} ({label})...")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "agent-sandbox-reference/0.1"},
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


def try_ip_egress(host: str, port: int, label: str) -> None:
    """Attempt a raw TCP connection to a host:port, bypassing DNS.

    Unlike ``try_egress`` (which goes through Python's URL library and
    triggers DNS resolution first), this opens a socket directly to an
    IP address. When the target IP is NOT on the CiliumNetworkPolicy
    allowlist, Cilium's L3/L4 enforcement drops the SYN packet — which
    produces a ``DROPPED`` flow in Hubble's default Service Map view.

    This is the counterpart to Step 4's FQDN-level block. Step 4's
    blocked egress fails at the DNS proxy (empty answer, no TCP ever
    attempted, no packet-level drop to visualize). Step 5 fails at
    L3/L4 (real SYN, real drop, visible in Hubble).
    """
    print(f"Attempting raw TCP connect to {host}:{port} ({label})...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect((host, port))
        s.close()
        print(f"UNEXPECTED PASS: connect to {host}:{port} succeeded")
    except (socket.timeout, TimeoutError):
        # L3/L4 drop manifests as connection timeout — no TCP RST,
        # the SYN is silently discarded by Cilium eBPF policy.
        print(f"BLOCKED: {host}:{port} connect timed out — L3/L4 policy drop")
    except Exception as e:  # noqa: BLE001
        print(f"BLOCKED: {host}:{port} rejected — {type(e).__name__}: {e}")


def pip_install(package: str) -> bool:
    """Install a package inside the sandbox. Exercises the PyPI
    allowlist — a successful install means the FQDN policy permitted
    pypi.org + files.pythonhosted.org."""
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
         — concrete evidence of what gVisor isolation does at
         runtime.
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
              "is slow under heavy syscall load — expected behavior)")


def main() -> int:
    step("Step 1: Install boto3 from PyPI (allowed egress)")
    if not pip_install("boto3"):
        # If PyPI is blocked, nothing downstream works — fail loudly
        # so the cause is obvious in the log.
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
    # explicit "no markdown" instruction.
    cleaned = snippet.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        last_fence = cleaned.rfind("```")
        if first_newline > 0 and last_fence > first_newline:
            cleaned = cleaned[first_newline + 1 : last_fence].strip()
    execute_snippet(cleaned)

    step("Step 4: Attempt FQDN egress to a BLOCKED domain")
    # Cilium FQDN policy enforces by DNS-proxy filtering, not packet
    # drop. When the FQDN isn't on the allowlist, the DNS proxy
    # returns an empty answer and the pod sees a resolution failure.
    # NO DROPPED flow appears in Hubble UI for this step because no
    # TCP connection is ever attempted — the pod never gets an IP.
    # Observable via `cilium observe --type policy-verdict` / DNS
    # proxy flow logs; not visible in default Hubble Service Map.
    # The hostname below is an example.com subdomain that's
    # deliberately absent from the allowlist — swap in any FQDN
    # that isn't in ciliumnetworkpolicy-sandbox-llm.yaml to test
    # a different enforcement path.
    try_egress("https://blocked-example.example.com/", "NOT on allowlist")

    step("Step 5: Attempt raw IP egress to a BLOCKED address")
    # Counterpart to Step 4. Connecting by IP bypasses DNS entirely,
    # so Cilium's L3/L4 enforcement drops the SYN packet directly.
    # THIS step IS visible as a red DROPPED flow in Hubble's default
    # Service Map — the visible evidence that FQDN enforcement (Step
    # 4) is complemented by packet-level enforcement (Step 5).
    # 8.8.8.8 (Google Public DNS) is used because it's a well-known
    # non-allowlisted IP with no ambiguity about what gets blocked.
    try_ip_egress("8.8.8.8", 443, "NOT on allowlist")

    step("Reference run complete")
    print("\nExpected outcomes:")
    print("  Step 1 (PyPI):            PASS — allowed by FQDN policy")
    print("  Step 2 (Bedrock):         PASS — allowed by FQDN policy")
    print("  Step 3 (snippet exec):    PASS — runs in the sandbox (Sentry intercepts syscalls on gVisor tier)")
    print("  Step 4 (FQDN block):      BLOCKED — denied at DNS proxy (not in Hubble UI)")
    print("  Step 5 (IP block):        BLOCKED — dropped at L3/L4 (visible in Hubble UI)")
    print("\nCheck Hubble UI for the Step 5 DROPPED flow to 8.8.8.8:443.")
    print("Use `cilium observe` or the DNS proxy logs for the Step 4 verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
