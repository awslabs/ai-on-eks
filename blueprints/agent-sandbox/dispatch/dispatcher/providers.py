"""Execution providers — one per axis.

A provider creates the per-session CR for its axis, resolves the
connection reference from CR status, and deletes what it created on
release or bind failure. All state lives in the CRs; the controllers
(agent-sandbox, ACK lambdamicrovms) own reconciliation.
"""

from __future__ import annotations

import logging
import time

from .models import (
    MANAGED_BY_LABEL,
    MANAGED_BY_VALUE,
    MICROVM_GROUP,
    MICROVM_VERSION,
    SANDBOX_GROUP,
    SANDBOX_VERSION,
    SESSION_LABEL,
    BindTimeoutError,
    Binding,
    Tier,
)

logger = logging.getLogger(__name__)


def _session_labels(session_id: str, tier: Tier) -> dict[str, str]:
    return {
        SESSION_LABEL: session_id,
        MANAGED_BY_LABEL: MANAGED_BY_VALUE,
        "agent-sandbox/tier": tier.name,
    }


def _poll(fn, timeout: int, interval: float = 2.0):
    """Poll fn() until it returns a truthy value or the deadline passes."""
    deadline = time.monotonic() + timeout
    while True:
        value = fn()
        if value:
            return value
        if time.monotonic() >= deadline:
            return None
        time.sleep(interval)


class SandboxClaimProvider:
    """In-cluster axis: SandboxClaim checked out from a SandboxWarmPool."""

    kind = "SandboxClaim"

    def __init__(self, dyn_client):
        self._api = dyn_client.resources.get(
            api_version=f"{SANDBOX_GROUP}/{SANDBOX_VERSION}", kind="SandboxClaim"
        )

    def bind(self, session_id: str, tier: Tier, timeout: int = 180) -> Binding:
        name = f"dispatch-{session_id[:16]}"
        manifest = {
            "apiVersion": f"{SANDBOX_GROUP}/{SANDBOX_VERSION}",
            "kind": "SandboxClaim",
            "metadata": {
                "name": name,
                "namespace": tier.namespace,
                "labels": _session_labels(session_id, tier),
            },
            "spec": {"warmPoolRef": {"name": tier.resource_name}},
        }
        self._api.create(body=manifest, namespace=tier.namespace)

        def _sandbox_name():
            claim = self._api.get(name=name, namespace=tier.namespace)
            status = getattr(claim, "status", None)
            sandbox = getattr(status, "sandbox", None) if status else None
            return getattr(sandbox, "name", None) if sandbox else None

        # Pod name == sandbox name (v1.0 guarantee). Never assume the
        # claim name: warm-adopted sandboxes carry pool-derived names.
        sandbox_name = _poll(_sandbox_name, timeout)
        if not sandbox_name:
            self.release(session_id, tier.namespace)
            raise BindTimeoutError(
                f"SandboxClaim {name} did not bind a sandbox within {timeout}s"
            )
        return Binding(
            session_id=session_id,
            tier=tier,
            kind=self.kind,
            name=name,
            namespace=tier.namespace,
            pod=sandbox_name,
        )

    def release(self, session_id: str, namespace: str) -> int:
        """Delete this session's claims by label; returns count deleted."""
        selector = f"{SESSION_LABEL}={session_id},{MANAGED_BY_LABEL}={MANAGED_BY_VALUE}"
        items = list(
            self._api.get(namespace=namespace, label_selector=selector).items or []
        )
        for item in items:
            self._api.delete(name=item.metadata.name, namespace=namespace)
        return len(items)


class MicrovmProvider:
    """Off-cluster axis: per-session Microvm launched from a MicrovmImage."""

    kind = "Microvm"

    def __init__(self, dyn_client, execution_role_arn: str,
                 max_duration_seconds: int = 3600,
                 idle_suspend_seconds: int = 300):
        self._api = dyn_client.resources.get(
            api_version=f"{MICROVM_GROUP}/{MICROVM_VERSION}", kind="Microvm"
        )
        self.execution_role_arn = execution_role_arn
        self.max_duration_seconds = max_duration_seconds
        self.idle_suspend_seconds = idle_suspend_seconds

    def bind(self, session_id: str, tier: Tier, timeout: int = 180) -> Binding:
        name = f"dispatch-{session_id[:16]}"
        manifest = {
            "apiVersion": f"{MICROVM_GROUP}/{MICROVM_VERSION}",
            "kind": "Microvm",
            "metadata": {
                "name": name,
                "namespace": tier.namespace,
                "labels": _session_labels(session_id, tier),
            },
            "spec": {
                "imageIdentifierRef": {"from": {"name": tier.resource_name}},
                "executionRoleARN": self.execution_role_arn,
                "maximumDurationInSeconds": self.max_duration_seconds,
                "idlePolicy": {
                    "maxIdleDurationSeconds": self.idle_suspend_seconds,
                    "autoResumeEnabled": True,
                },
            },
        }
        self._api.create(body=manifest, namespace=tier.namespace)

        def _endpoint():
            vm = self._api.get(name=name, namespace=tier.namespace)
            status = getattr(vm, "status", None)
            if status and getattr(status, "state", None) == "Running":
                return getattr(status, "endpoint", None)
            return None

        endpoint = _poll(_endpoint, timeout)
        if not endpoint:
            self.release(session_id, tier.namespace)
            raise BindTimeoutError(
                f"Microvm {name} did not reach Running with an endpoint within {timeout}s"
            )
        return Binding(
            session_id=session_id,
            tier=tier,
            kind=self.kind,
            name=name,
            namespace=tier.namespace,
            endpoint=endpoint,
        )

    def release(self, session_id: str, namespace: str) -> int:
        selector = f"{SESSION_LABEL}={session_id},{MANAGED_BY_LABEL}={MANAGED_BY_VALUE}"
        items = list(
            self._api.get(namespace=namespace, label_selector=selector).items or []
        )
        for item in items:
            self._api.delete(name=item.metadata.name, namespace=namespace)
        return len(items)
