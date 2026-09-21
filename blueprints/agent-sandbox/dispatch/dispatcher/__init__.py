"""Sandbox dispatch — session→execution-target binding for agent sandboxes.

Two axes behind one interface: in-cluster SandboxClaim tiers (runc /
gvisor / kata-fc via labeled SandboxWarmPools) and the off-cluster
Lambda MicroVM tier (labeled MicrovmImages via the ACK controller).
See DESIGN.md for the architecture and its anti-goals.
"""

from .core import Dispatcher, PreBindVeto
from .models import (
    Binding,
    BindTimeoutError,
    Tier,
    TierNotFoundError,
)

__all__ = [
    "Dispatcher",
    "PreBindVeto",
    "Binding",
    "BindTimeoutError",
    "Tier",
    "TierNotFoundError",
]
