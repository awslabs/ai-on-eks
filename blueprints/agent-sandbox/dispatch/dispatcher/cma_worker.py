"""CMA work-queue loop — the dispatcher as a Claude Managed Agents worker.

Implements the self-hosted-sandbox worker shape from the upstream
agent-sandbox CMA use case: Anthropic hosts the agent loop and model;
this loop polls the environment work queue and gives each session an
isolated execution target via the dispatcher — one session per sandbox,
tier chosen by dispatch (in-cluster today; the microvm tier applies
unchanged once its worker image speaks the CMA harness protocol).

This is a *hook consumer*, not new dispatcher machinery: binding rides
``Dispatcher.bind`` so admission (`pre_bind`) and attribution
(`post_bind`) hooks apply to CMA sessions like any other.

Dependencies are injected so the loop unit-tests without the anthropic
SDK or a cluster:
- ``queue``: anything with ``poll(env_id, block_ms=...) -> item | None``
  (the anthropic SDK's ``client.beta.environments.work`` fits; pin the
  ``managed-agents-2026-04-01`` beta — the SDK sets it automatically)
- ``post_work``: delivers the (session_id, work_id) binding to the
  worker pod. Default implementation resolves the pod IP and POSTs to
  the harness port, matching the upstream reference dispatcher.

Worker pods must run the Anthropic worker harness (which pulls tool
calls and posts results back) — see the CMA templates, not the
python-runtime dispatch workers. Session teardown: the claim's
lifecycle (TTL / shutdownPolicy on the CMA template) owns routine
teardown; ``release_session`` exists for explicit reaping.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Callable

from .core import Dispatcher, PreBindVeto
from .models import Binding

logger = logging.getLogger(__name__)

HARNESS_PORT = 8080  # Anthropic worker harness listen port


def default_post_work(dyn_client, binding: Binding, session_id: str, work_id: str,
                      port: int = HARNESS_PORT, timeout: int = 10) -> None:
    """POST the session binding to the worker pod's harness endpoint.

    Resolves the pod IP directly (the per-claim Service DNS record is
    too fresh to resolve reliably on first contact — same finding as
    the upstream reference dispatcher).
    """
    pods = dyn_client.resources.get(api_version="v1", kind="Pod")
    pod = pods.get(name=binding.pod, namespace=binding.namespace)
    pod_ip = pod.status.podIP
    req = urllib.request.Request(
        f"http://{pod_ip}:{port}/",
        data=json.dumps({"session_id": session_id, "work_id": work_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=timeout)


class CMAWorker:
    def __init__(
        self,
        dispatcher: Dispatcher,
        queue: Any,
        env_id: str,
        tier: str = "gvisor",
        post_work: Callable[[Binding, str, str], None] | None = None,
        poll_block_ms: int = 900,
        error_backoff_s: float = 5.0,
    ):
        self.dispatcher = dispatcher
        self.queue = queue
        self.env_id = env_id
        self.tier = tier
        self.post_work = post_work
        self.poll_block_ms = poll_block_ms
        self.error_backoff_s = error_backoff_s

    def run_once(self) -> Binding | None:
        """One poll→bind→deliver cycle. Returns the Binding, or None if
        the queue was empty. Raises nothing on work-item failure: an
        unacknowledged item is redelivered by the queue, so failures
        log-and-return rather than crash the loop."""
        item = self.queue.poll(self.env_id, block_ms=self.poll_block_ms)
        if item is None:
            return None

        session_id = item.data.id
        work_id = item.id
        try:
            binding = self.dispatcher.bind(session_id, self.tier)
        except PreBindVeto as veto:
            logger.warning("CMA session %s vetoed: %s", session_id, veto)
            return None
        except Exception:
            logger.exception("bind failed for CMA session %s; item will redeliver", session_id)
            return None

        try:
            if self.post_work is not None:
                self.post_work(binding, session_id, work_id)
        except Exception:
            # Sandbox exists but the harness handoff failed — release so
            # the redelivered item gets a fresh bind instead of a
            # half-initialized worker.
            logger.exception("work handoff failed for session %s; releasing", session_id)
            self.dispatcher.release(session_id)
            return None

        logger.info("CMA session %s dispatched to %s (%s)",
                    session_id, binding.pod or binding.endpoint, self.tier)
        return binding

    def release_session(self, session_id: str) -> int:
        """Explicit reap; routine teardown belongs to claim lifecycle."""
        return self.dispatcher.release(session_id)

    def run_forever(self) -> None:
        logger.info("CMA worker loop: env=%s tier=%s", self.env_id, self.tier)
        while True:
            try:
                self.run_once()
            except Exception:
                logger.exception("poll cycle failed; backing off %.1fs", self.error_backoff_s)
                time.sleep(self.error_backoff_s)
