"""Dispatcher — session→execution-target binding with lifecycle hooks.

Thin by design: discovery via the label registry, CR creation via the
providers, teardown by label selector. No reconcile loop, no state that
`kubectl get -l agent-sandbox/session-id=<id>` can't show.
"""

from __future__ import annotations

import logging
from typing import Callable

from .models import Binding, Tier
from .providers import MicrovmProvider, SandboxClaimProvider
from .registry import TierRegistry

logger = logging.getLogger(__name__)

Hook = Callable[[Binding], None]


class PreBindVeto(Exception):
    """Raised by a pre_bind hook to reject a binding (admission point)."""


class Dispatcher:
    def __init__(
        self,
        dyn_client,
        namespace: str = "agent-sandboxes",
        microvm_execution_role_arn: str | None = None,
        microvm_ingress_connector_arns: list[str] | None = None,
        microvm_egress_connector_arns: list[str] | None = None,
    ):
        self.registry = TierRegistry(dyn_client, namespace)
        self.namespace = namespace
        self._providers = {
            "SandboxWarmPool": SandboxClaimProvider(dyn_client),
        }
        if microvm_execution_role_arn:
            self._providers["MicrovmImage"] = MicrovmProvider(
                dyn_client,
                microvm_execution_role_arn,
                ingress_connector_arns=microvm_ingress_connector_arns,
                egress_connector_arns=microvm_egress_connector_arns,
            )
        self._hooks: dict[str, list[Hook]] = {
            "pre_bind": [], "post_bind": [], "pre_release": [], "post_release": [],
        }

    # -- hooks ---------------------------------------------------------
    def on(self, event: str, hook: Hook) -> None:
        """Register a lifecycle hook (pre_bind may veto; others fail open)."""
        self._hooks[event].append(hook)

    def _fire(self, event: str, binding: Binding) -> None:
        for hook in self._hooks[event]:
            try:
                hook(binding)
            except PreBindVeto:
                raise
            except Exception:
                # Fail open: hooks other than pre_bind vetoes never block
                # the session path.
                logger.exception("hook %s failed (ignored)", event)

    # -- session ops ----------------------------------------------------
    def tiers(self) -> dict[str, Tier]:
        return self.registry.tiers()

    def bind(self, session_id: str, tier_name: str, timeout: int = 180) -> Binding:
        tier = self.registry.resolve(tier_name)
        provider = self._providers.get(tier.kind)
        if provider is None:
            raise LookupError(
                f"tier '{tier_name}' is served by {tier.kind}, but no provider is "
                "configured for it (Microvm tiers need microvm_execution_role_arn)"
            )
        intent = Binding(
            session_id=session_id, tier=tier, kind=provider.kind,
            name="", namespace=tier.namespace,
        )
        # pre_bind is the admission point — a PreBindVeto here rejects
        # the session before anything is created.
        self._fire("pre_bind", intent)
        binding = provider.bind(session_id, tier, timeout=timeout)
        self._fire("post_bind", binding)
        logger.info(
            "bound session=%s tier=%s (%s) -> %s",
            session_id, tier_name, tier.axis, binding.pod or binding.endpoint,
        )
        return binding

    def release(self, session_id: str) -> int:
        """Release all of a session's targets across providers. Idempotent."""
        marker = Binding(
            session_id=session_id,
            tier=Tier(name="*", kind="*", resource_name="*", namespace=self.namespace),
            kind="*", name="*", namespace=self.namespace,
        )
        self._fire("pre_release", marker)
        deleted = 0
        for provider in self._providers.values():
            deleted += provider.release(session_id, self.namespace)
        self._fire("post_release", marker)
        logger.info("released session=%s (%d resource(s))", session_id, deleted)
        return deleted
