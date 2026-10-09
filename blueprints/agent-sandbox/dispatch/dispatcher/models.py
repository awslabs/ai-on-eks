"""Data model for the sandbox dispatcher.

The dispatcher holds no state of its own — these types describe what
the cluster already knows (labeled capacity resources, per-session CRs).
See ../README.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Label carried by capacity resources (SandboxWarmPool, MicrovmImage)
# to register as a dispatchable tier, and stamped on everything the
# dispatcher creates.
TIER_LABEL = "agent-sandbox/tier"
SESSION_LABEL = "agent-sandbox/session-id"
MANAGED_BY_LABEL = "agent-sandbox/managed-by"
MANAGED_BY_VALUE = "dispatch"
# Optional annotation on capacity resources; higher wins when multiple
# resources carry the same tier label (blue/green template rollouts).
PRIORITY_ANNOTATION = "agent-sandbox/priority"

SANDBOX_GROUP = "extensions.agents.x-k8s.io"
SANDBOX_VERSION = "v1beta1"
MICROVM_GROUP = "lambdamicrovms.services.k8s.aws"
MICROVM_VERSION = "v1alpha1"


@dataclass(frozen=True)
class Tier:
    """A dispatchable execution tier discovered from the cluster."""

    name: str                 # the agent-sandbox/tier label value
    kind: str                 # SandboxWarmPool | MicrovmImage
    resource_name: str        # name of the capacity resource
    namespace: str
    priority: int = 0

    @property
    def axis(self) -> str:
        return "in-cluster" if self.kind == "SandboxWarmPool" else "off-cluster"


@dataclass
class Binding:
    """A session bound to an execution target."""

    session_id: str
    tier: Tier
    kind: str                 # SandboxClaim | Microvm
    name: str                 # name of the per-session CR
    namespace: str
    # Connection reference: pod name (in-cluster; == sandbox name, a
    # v1.0 API guarantee) or the Microvm endpoint URL (off-cluster).
    pod: str | None = None
    endpoint: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class TierNotFoundError(LookupError):
    def __init__(self, tier: str, available: list[str]):
        super().__init__(
            f"tier '{tier}' not found; available tiers: {sorted(available) or '(none)'}"
        )
        self.tier = tier
        self.available = available


class BindTimeoutError(TimeoutError):
    pass
