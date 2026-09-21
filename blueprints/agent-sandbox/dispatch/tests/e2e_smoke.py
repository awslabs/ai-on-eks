"""Dispatcher E2E smoke — requires a LIVE cluster with the dispatch
worker manifests applied (see ../README.md steps 1-2).

Measures two phases per bind and prints a results table:
  bind_s   dispatcher.bind() wall time (claim create -> status.sandbox.name)
  ready_s  additional wait for the backing pod to reach Ready

Validates: two-axis registry discovery, warm vs cold bind (pool
capacity exhaustion), warm/cold pod-naming semantics, the session-label
operational surface, pre_bind veto, idempotent release with zero
orphans.

Run: python tests/e2e_smoke.py   (from the dispatch/ directory)
Reference numbers (fresh Standard EKS rig, Sep 2026): warm bind ~0.8s
runtime-independent; cold ~3s on gvisor; pod-ready +2-3s.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kubernetes import config, dynamic
from kubernetes.client import api_client

from dispatcher import Dispatcher, PreBindVeto

NS = os.environ.get("DISPATCH_NAMESPACE", "agent-sandboxes")


def wait_pod_ready(pod: str, timeout: int = 300) -> float:
    t0 = time.monotonic()
    r = subprocess.run(
        ["kubectl", "-n", NS, "wait", "--for=condition=Ready",
         f"pod/{pod}", f"--timeout={timeout}s"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"pod {pod} not Ready: {r.stderr.strip()[-200:]}")
    return time.monotonic() - t0


def kubectl_out(*args: str) -> str:
    return subprocess.run(
        ["kubectl", *args], capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    config.load_kube_config()
    d = Dispatcher(dynamic.DynamicClient(api_client.ApiClient()), namespace=NS)
    results: dict[str, tuple] = {}

    tiers = d.tiers()
    print(f"[1] tiers: { {k: v.axis for k, v in tiers.items()} }")
    assert "gvisor" in tiers and "runc" in tiers, "apply worker manifests first"

    t0 = time.monotonic()
    b1 = d.bind("smoke-warm-001", "gvisor")
    warm_bind = time.monotonic() - t0
    warm_ready = wait_pod_ready(b1.pod)
    print(f"[2] WARM gvisor bind={warm_bind:.2f}s ready+={warm_ready:.2f}s pod={b1.pod}")
    results["warm_gvisor"] = (round(warm_bind, 2), round(warm_ready, 2))

    t0 = time.monotonic()
    b2 = d.bind("smoke-cold-002", "gvisor", timeout=300)
    cold_bind = time.monotonic() - t0
    cold_ready = wait_pod_ready(b2.pod)
    print(f"[3] COLD gvisor bind={cold_bind:.2f}s ready+={cold_ready:.2f}s pod={b2.pod}")
    results["cold_gvisor"] = (round(cold_bind, 2), round(cold_ready, 2))
    assert b2.pod.startswith("dispatch-smoke-cold"), "cold pod should be claim-named"
    assert not b1.pod.startswith("dispatch-smoke-warm"), "warm pod should be pool-named"

    out = kubectl_out("-n", NS, "get", "sandboxclaims",
                      "-l", "agent-sandbox/session-id=smoke-warm-001", "-o", "name")
    assert out, "session-label surface broken"
    print(f"[4] label surface OK: {out}")

    def deny(binding):
        raise PreBindVeto("smoke: deny all")

    d.on("pre_bind", deny)
    try:
        d.bind("smoke-veto-003", "gvisor")
        raise AssertionError("veto did not block")
    except PreBindVeto:
        print("[5] pre_bind veto OK")
    d._hooks["pre_bind"].clear()

    for sid in ("smoke-warm-001", "smoke-cold-002"):
        print(f"[6] release {sid}: {d.release(sid)} resource(s)")
    assert d.release("smoke-warm-001") == 0, "release not idempotent"
    left = kubectl_out("-n", NS, "get", "sandboxclaims",
                       "-l", "agent-sandbox/managed-by=dispatch", "-o", "name")
    assert not left, f"orphaned claims: {left}"
    print("[6] released, idempotent, zero orphans")

    print("\n=== RESULTS (bind_s, ready_s) ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("PASS: dispatcher E2E smoke")


if __name__ == "__main__":
    main()
