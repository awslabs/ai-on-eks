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


def _release_by_selector(api, session_id: str, namespace: str) -> int:
    """Delete a session's CRs by label; returns count newly deleted.

    Resources already terminating (deletionTimestamp set) are skipped —
    controllers with finalizers (e.g. ACK while the VM shuts down) keep
    the CR visible for a while, and re-deleting it isn't a release.
    """
    selector = f"{SESSION_LABEL}={session_id},{MANAGED_BY_LABEL}={MANAGED_BY_VALUE}"
    items = list(api.get(namespace=namespace, label_selector=selector).items or [])
    deleted = 0
    for item in items:
        if getattr(item.metadata, "deletionTimestamp", None):
            continue
        api.delete(name=item.metadata.name, namespace=namespace)
        deleted += 1
    return deleted


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
        return _release_by_selector(self._api, session_id, namespace)


class MicrovmProvider:
    """Off-cluster axis: per-session Microvm launched from a MicrovmImage."""

    kind = "Microvm"

    def __init__(self, dyn_client, execution_role_arn: str,
                 max_duration_seconds: int = 3600,
                 idle_suspend_seconds: int = 300,
                 suspended_retention_seconds: int = 86400,
                 ingress_connector_arns: list[str] | None = None,
                 egress_connector_arns: list[str] | None = None):
        self._api = dyn_client.resources.get(
            api_version=f"{MICROVM_GROUP}/{MICROVM_VERSION}", kind="Microvm"
        )
        self.execution_role_arn = execution_role_arn
        self.max_duration_seconds = max_duration_seconds
        self.idle_suspend_seconds = idle_suspend_seconds
        # Required by the API alongside maxIdleDurationSeconds — bounds
        # how long a suspended VM is retained before termination.
        self.suspended_retention_seconds = suspended_retention_seconds
        # Network connectors are Lambda-managed ARNs, e.g.
        # arn:aws:lambda:<region>:aws:network-connector:aws-network-connector:ALL_INGRESS
        # Without an ingress connector, endpoint traffic can't reach the
        # worker, so binds would succeed but the session would be unusable.
        self.ingress_connector_arns = ingress_connector_arns or []
        self.egress_connector_arns = egress_connector_arns or []

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
                    "suspendedDurationSeconds": self.suspended_retention_seconds,
                },
            },
        }
        if self.ingress_connector_arns:
            manifest["spec"]["ingressNetworkConnectors"] = self.ingress_connector_arns
        if self.egress_connector_arns:
            manifest["spec"]["egressNetworkConnectors"] = self.egress_connector_arns
        self._api.create(body=manifest, namespace=tier.namespace)

        def _endpoint():
            vm = self._api.get(name=name, namespace=tier.namespace)
            status = getattr(vm, "status", None)
            # API lifecycle states are upper case (PENDING -> RUNNING).
            if status and str(getattr(status, "state", "")).upper() == "RUNNING":
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
        return _release_by_selector(self._api, session_id, namespace)
