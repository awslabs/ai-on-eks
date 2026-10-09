"""Label-based tier registry — the cluster is the registry.

Discovery is a namespaced label list at bind time. No cache, no watch,
no config file: adding a tier is `kubectl apply` of a labeled
SandboxWarmPool (in-cluster) or MicrovmImage (off-cluster).
"""

from __future__ import annotations

import logging

from .models import (
    MICROVM_GROUP,
    MICROVM_VERSION,
    PRIORITY_ANNOTATION,
    SANDBOX_GROUP,
    SANDBOX_VERSION,
    TIER_LABEL,
    Tier,
    TierNotFoundError,
)

logger = logging.getLogger(__name__)


class TierRegistry:
    """Discovers dispatchable tiers from labeled capacity resources."""

    def __init__(self, dyn_client, namespace: str):
        self._dyn = dyn_client
        self.namespace = namespace

    def _list(self, group: str, version: str, kind: str) -> list:
        try:
            api = self._dyn.resources.get(
                api_version=f"{group}/{version}", kind=kind
            )
        except Exception:  # CRD not installed (e.g. ACK addon disabled)
            logger.debug("kind %s/%s %s not served; skipping", group, version, kind)
            return []
        result = api.get(namespace=self.namespace, label_selector=TIER_LABEL)
        return list(result.items or [])

    def tiers(self) -> dict[str, Tier]:
        """All tiers visible in the namespace, deduped by priority."""
        found: dict[str, Tier] = {}
        for kind in ("SandboxWarmPool", "MicrovmImage"):
            group, version = (
                (SANDBOX_GROUP, SANDBOX_VERSION)
                if kind == "SandboxWarmPool"
                else (MICROVM_GROUP, MICROVM_VERSION)
            )
            for item in self._list(group, version, kind):
                labels = item.metadata.labels or {}
                annotations = item.metadata.annotations or {}
                tier_name = labels.get(TIER_LABEL)
                if not tier_name:
                    continue
                try:
                    priority = int(annotations.get(PRIORITY_ANNOTATION, "0"))
                except ValueError:
                    priority = 0
                candidate = Tier(
                    name=tier_name,
                    kind=kind,
                    resource_name=item.metadata.name,
                    namespace=self.namespace,
                    priority=priority,
                )
                current = found.get(tier_name)
                # Higher priority wins; tie broken by resource name for
                # deterministic selection.
                if (
                    current is None
                    or candidate.priority > current.priority
                    or (
                        candidate.priority == current.priority
                        and candidate.resource_name < current.resource_name
                    )
                ):
                    found[tier_name] = candidate
        return found

    def resolve(self, tier_name: str) -> Tier:
        tiers = self.tiers()
        if tier_name not in tiers:
            raise TierNotFoundError(tier_name, list(tiers))
        return tiers[tier_name]
