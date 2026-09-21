"""Unit tests for the dispatcher — no cluster, no kubernetes package.

The dispatcher takes a dynamic client as a constructor argument and
never imports kubernetes itself, so everything here runs against small
stubs: registry dedup/priority semantics, degradation when the ACK CRDs
aren't installed, and hook veto / fail-open behavior.

Run: python3 -m pytest tests/test_unit.py  (or python3 tests/test_unit.py)
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dispatcher.core import Dispatcher, PreBindVeto
from dispatcher.models import Binding, Tier, TierNotFoundError
from dispatcher.registry import TierRegistry


# --- stubs -------------------------------------------------------------------

class _Obj:
    def __init__(self, name, labels=None, annotations=None):
        self.metadata = types.SimpleNamespace(
            name=name, labels=labels or {}, annotations=annotations or {}
        )


class _ListResult:
    def __init__(self, items):
        self.items = items


class _API:
    def __init__(self, items):
        self._items = items

    def get(self, namespace=None, label_selector=None, name=None):
        return _ListResult(self._items)


class _Resources:
    def __init__(self, by_kind):
        self.by_kind = by_kind

    def get(self, api_version=None, kind=None):
        if kind not in self.by_kind:
            raise Exception("CRD not served")
        return _API(self.by_kind[kind])


class _Dyn:
    def __init__(self, by_kind):
        self.resources = _Resources(by_kind)


class _Provider:
    kind = "SandboxClaim"

    def bind(self, sid, tier, timeout=180):
        return Binding(session_id=sid, tier=tier, kind=self.kind,
                       name="x", namespace="ns", pod="pod-x")

    def release(self, sid, ns):
        return 1


def _pools():
    return [
        _Obj("pool-blue", {"agent-sandbox/tier": "gvisor"},
             {"agent-sandbox/priority": "10"}),
        _Obj("pool-green", {"agent-sandbox/tier": "gvisor"}),
        _Obj("pool-kata", {"agent-sandbox/tier": "kata-fc"}),
    ]


def _dispatcher(reg):
    d = Dispatcher.__new__(Dispatcher)
    d.registry, d.namespace = reg, "ns"
    d._providers = {"SandboxWarmPool": _Provider(), "MicrovmImage": _Provider()}
    d._hooks = {k: [] for k in
                ("pre_bind", "post_bind", "pre_release", "post_release")}
    return d


# --- registry ----------------------------------------------------------------

def test_registry_dedup_priority_and_axes():
    reg = TierRegistry(_Dyn({
        "SandboxWarmPool": _pools(),
        "MicrovmImage": [_Obj("worker-img", {"agent-sandbox/tier": "microvm"})],
    }), "ns")
    tiers = reg.tiers()
    assert set(tiers) == {"gvisor", "kata-fc", "microvm"}
    assert tiers["gvisor"].resource_name == "pool-blue"  # priority wins
    assert tiers["microvm"].axis == "off-cluster"
    assert tiers["kata-fc"].axis == "in-cluster"


def test_registry_error_lists_available_tiers():
    reg = TierRegistry(_Dyn({"SandboxWarmPool": _pools()}), "ns")
    try:
        reg.resolve("no-such-tier")
        raise AssertionError("expected TierNotFoundError")
    except TierNotFoundError as e:
        assert "kata-fc" in str(e) and e.tier == "no-such-tier"


def test_registry_degrades_without_ack_crds():
    reg = TierRegistry(_Dyn({"SandboxWarmPool": _pools()}), "ns")
    assert set(reg.tiers()) == {"gvisor", "kata-fc"}


# --- dispatcher hooks ---------------------------------------------------------

def test_bind_fires_hooks_in_order():
    reg = TierRegistry(_Dyn({"SandboxWarmPool": _pools()}), "ns")
    d = _dispatcher(reg)
    events = []
    d.on("pre_bind", lambda b: events.append(("pre", b.tier.name)))
    d.on("post_bind", lambda b: events.append(("post", b.pod)))
    b = d.bind("s1", "gvisor")
    assert b.pod == "pod-x"
    assert events == [("pre", "gvisor"), ("post", "pod-x")]


def test_non_veto_hook_failure_is_fail_open():
    reg = TierRegistry(_Dyn({"SandboxWarmPool": _pools()}), "ns")
    d = _dispatcher(reg)
    d.on("post_release", lambda b: 1 / 0)
    assert d.release("s1") == 2  # both providers report 1


def test_pre_bind_veto_blocks():
    reg = TierRegistry(_Dyn({"SandboxWarmPool": _pools()}), "ns")
    d = _dispatcher(reg)

    def deny(binding):
        raise PreBindVeto("denied")

    d.on("pre_bind", deny)
    try:
        d.bind("s2", "gvisor")
        raise AssertionError("expected PreBindVeto")
    except PreBindVeto:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok {fn.__name__}")
    print(f"{len(fns)} unit tests passed")
