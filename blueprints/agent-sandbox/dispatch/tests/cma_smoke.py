"""CMA worker-loop live smoke — requires a LIVE cluster with the
dispatch worker manifests applied (see ../README.md steps 1-2).

Drives CMAWorker.run_once() with a stub queue against the real
dispatcher and cluster, validating the poll -> bind -> deliver cycle
plus the loop's failure semantics:
  [1] one queue item -> real bind -> work executes in the sandbox:
      POST /execute on the runtime's API runs a command that echoes
      the session binding; asserted on exit_code and stdout content
      (kubectl exec is only the network path — pod IPs aren't routable
      from outside the VPC)
  [2] handoff failure -> session released (fresh bind on redelivery)
  [3] pre_bind veto -> item dropped, nothing created
  [4] empty queue -> clean no-op
  [5] explicit reap -> zero session resources left

Run: python tests/cma_smoke.py   (from the dispatch/ directory)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kubernetes import config, dynamic
from kubernetes.client import api_client

from dispatcher import Dispatcher, PreBindVeto
from dispatcher.cma_worker import CMAWorker

NS = os.environ.get("DISPATCH_NAMESPACE", "agent-sandboxes")
RUNTIME_PORT = 8888  # python-runtime worker port (stands in for the harness)


class StubQueue:
    """Delivers a fixed list of work items, then reports empty."""

    def __init__(self, items):
        self._items = list(items)

    def poll(self, env_id, block_ms=0):
        return self._items.pop(0) if self._items else None


def work_item(session_id: str, work_id: str):
    return SimpleNamespace(id=work_id, data=SimpleNamespace(id=session_id))


def exec_post_work(binding, session_id, work_id):
    """Deliver work to the dispatched sandbox and verify it executed.

    Real handoff against the python-runtime's actual API: POST /execute
    (schema: {"command"} -> {"stdout","stderr","exit_code"}) runs a
    command inside the sandbox that echoes the session binding. Success
    requires exit_code == 0 AND the session_id in stdout — proof the
    work ran in the sandbox, not just that a socket answered.

    (kubectl exec is only the network path to the pod — from outside
    the VPC pod IPs aren't routable. The request itself goes through
    the runtime's HTTP API exactly as the in-cluster reference
    dispatcher's harness handoff does.)
    """
    payload = json.dumps(
        {"command": f"echo dispatched session={session_id} work={work_id}"}
    )
    code = (
        "import urllib.request, json, sys\n"
        f"r = urllib.request.urlopen(urllib.request.Request("
        f"'http://localhost:{RUNTIME_PORT}/execute', data={payload!r}.encode(), "
        "headers={'Content-Type': 'application/json'}), timeout=15)\n"
        "print(r.read().decode())\n"
    )
    r = subprocess.run(
        ["kubectl", "-n", NS, "exec", binding.pod, "-c", "runtime",
         "--", "python", "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        raise RuntimeError(f"handoff failed: {r.stderr.strip()[-200:]}")
    result = json.loads(r.stdout)
    if result["exit_code"] != 0 or f"session={session_id}" not in result["stdout"]:
        raise RuntimeError(f"work did not execute in sandbox: {result}")
    print(f"    executed in sandbox: {result['stdout'].strip()!r} exit={result['exit_code']}")


def main() -> None:
    config.load_kube_config()
    d = Dispatcher(dynamic.DynamicClient(api_client.ApiClient()), namespace=NS)

    # [1] poll -> bind -> deliver
    w = CMAWorker(d, StubQueue([work_item("cma-smoke-001", "work-1")]),
                  env_id="env-smoke", tier="gvisor", post_work=exec_post_work)
    b = w.run_once()
    assert b is not None and b.pod, "expected a live binding"
    print(f"[1] poll->bind->deliver OK: session=cma-smoke-001 pod={b.pod}")

    # [2] handoff failure releases the session
    def broken_handoff(binding, session_id, work_id):
        raise RuntimeError("simulated harness failure")

    w2 = CMAWorker(d, StubQueue([work_item("cma-smoke-002", "work-2")]),
                   env_id="env-smoke", tier="gvisor", post_work=broken_handoff)
    assert w2.run_once() is None
    left = subprocess.run(
        ["kubectl", "-n", NS, "get", "sandboxclaims",
         "-l", "agent-sandbox/session-id=cma-smoke-002", "-o", "name"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert not left, f"failed handoff left resources: {left}"
    print("[2] handoff failure -> session released OK")

    # [3] veto drops the item without creating anything
    def deny(binding):
        raise PreBindVeto("cma smoke: deny")

    d.on("pre_bind", deny)
    w3 = CMAWorker(d, StubQueue([work_item("cma-smoke-003", "work-3")]),
                   env_id="env-smoke", tier="gvisor", post_work=exec_post_work)
    assert w3.run_once() is None
    d._hooks["pre_bind"].clear()
    print("[3] pre_bind veto OK")

    # [4] empty queue is a clean no-op
    assert CMAWorker(d, StubQueue([]), env_id="env-smoke").run_once() is None
    print("[4] empty queue no-op OK")

    # [5] explicit reap
    assert w.release_session("cma-smoke-001") == 1
    assert w.release_session("cma-smoke-001") == 0
    print("[5] explicit reap, idempotent OK")

    print("\nPASS: CMA worker loop live smoke")


if __name__ == "__main__":
    main()
