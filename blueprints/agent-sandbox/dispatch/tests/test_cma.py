"""Unit tests for the CMA worker loop — no cluster, no anthropic SDK.

The loop's dependencies (queue, dispatcher, post_work) are injected, so
these tests exercise the poll→bind→deliver cycle and its failure
semantics with stubs: empty queue, successful dispatch, pre_bind veto,
bind failure (item left for redelivery), and handoff failure (session
released so redelivery gets a fresh bind).

Run: python3 -m pytest tests/test_cma.py  (or python3 tests/test_cma.py)
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dispatcher.cma_worker import CMAWorker
from dispatcher.core import PreBindVeto
from dispatcher.models import Binding, Tier


# --- stubs -------------------------------------------------------------------

def _work_item(session_id: str, work_id: str):
    return types.SimpleNamespace(id=work_id, data=types.SimpleNamespace(id=session_id))


class _Queue:
    def __init__(self, items):
        self._items = list(items)
        self.polls = 0

    def poll(self, env_id, block_ms=900):
        self.polls += 1
        return self._items.pop(0) if self._items else None


class _StubDispatcher:
    def __init__(self, fail_bind=False, veto=False):
        self.fail_bind = fail_bind
        self.veto = veto
        self.bound: list[str] = []
        self.released: list[str] = []

    def bind(self, session_id, tier, timeout=180):
        if self.veto:
            raise PreBindVeto("policy")
        if self.fail_bind:
            raise RuntimeError("no capacity")
        self.bound.append(session_id)
        return Binding(
            session_id=session_id,
            tier=Tier(name=tier, kind="SandboxWarmPool", resource_name="p", namespace="ns"),
            kind="SandboxClaim", name=f"dispatch-{session_id}", namespace="ns",
            pod=f"pod-{session_id}",
        )

    def release(self, session_id):
        self.released.append(session_id)
        return 1


# --- tests -------------------------------------------------------------------

def test_empty_queue_returns_none():
    w = CMAWorker(_StubDispatcher(), _Queue([]), env_id="env-1")
    assert w.run_once() is None


def test_successful_dispatch_binds_and_delivers():
    delivered = []
    d = _StubDispatcher()
    w = CMAWorker(
        d, _Queue([_work_item("sess-a", "work-1")]), env_id="env-1", tier="gvisor",
        post_work=lambda b, sid, wid: delivered.append((b.pod, sid, wid)),
    )
    b = w.run_once()
    assert b is not None and b.pod == "pod-sess-a"
    assert d.bound == ["sess-a"]
    assert delivered == [("pod-sess-a", "sess-a", "work-1")]
    assert d.released == []


def test_veto_drops_item_without_binding():
    d = _StubDispatcher(veto=True)
    w = CMAWorker(d, _Queue([_work_item("sess-b", "work-2")]), env_id="env-1")
    assert w.run_once() is None
    assert d.bound == [] and d.released == []


def test_bind_failure_leaves_item_for_redelivery():
    d = _StubDispatcher(fail_bind=True)
    w = CMAWorker(d, _Queue([_work_item("sess-c", "work-3")]), env_id="env-1")
    assert w.run_once() is None      # logged, not raised — loop survives
    assert d.released == []          # nothing was created, nothing to reap


def test_handoff_failure_releases_session():
    d = _StubDispatcher()

    def broken_post_work(binding, sid, wid):
        raise ConnectionError("harness unreachable")

    w = CMAWorker(d, _Queue([_work_item("sess-d", "work-4")]), env_id="env-1",
                  post_work=broken_post_work)
    assert w.run_once() is None
    assert d.bound == ["sess-d"]
    assert d.released == ["sess-d"]  # freed so redelivery binds fresh


def test_release_session_delegates():
    d = _StubDispatcher()
    w = CMAWorker(d, _Queue([]), env_id="env-1")
    assert w.release_session("sess-e") == 1
    assert d.released == ["sess-e"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} CMA tests passed")
