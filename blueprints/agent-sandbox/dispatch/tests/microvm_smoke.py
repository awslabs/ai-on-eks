"""MicroVM tier E2E smoke — requires a LIVE cluster with the ACK
lambdamicrovms addon (enable_ack_lambdamicrovms=true), a reconciled
MicrovmImage carrying the agent-sandbox/tier label (see
../../microvm/README.md), and AWS credentials with
lambda:CreateMicrovmAuthToken.

Validates the off-cluster half of the dispatch pattern end to end:
  [1] registry discovers the microvm tier from the labeled MicrovmImage
  [2] dispatcher.bind() creates a Microvm CR, ACK runs the VM, the
      binding returns status.endpoint (bind_s = create -> RUNNING+endpoint)
  [3] the endpoint answers through the Lambda proxy with a port-scoped
      auth token (X-aws-proxy-auth) — worker identity JSON on :8888
  [4] release deletes by session label; zero orphaned Microvm CRs

Env (defaults for a fresh rig):
  MICROVM_EXEC_ROLE_ARN   execution role passed to each Microvm
  AWS_REGION              used to build managed connector ARNs
  DISPATCH_NAMESPACE      default agent-sandboxes

Run: python tests/microvm_smoke.py   (from the dispatch/ directory)
Requires: kubernetes, boto3 (with the lambda-microvms service model).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import boto3
from kubernetes import config, dynamic
from kubernetes.client import api_client

from dispatcher import Dispatcher
from dispatcher.models import MICROVM_GROUP, MICROVM_VERSION

NS = os.environ.get("DISPATCH_NAMESPACE", "agent-sandboxes")
REGION = os.environ.get("AWS_REGION", "us-west-2")
EXEC_ROLE = os.environ.get(
    "MICROVM_EXEC_ROLE_ARN",
    "arn:aws:iam::893848774378:role/agent-sandbox-microvm-exec",
)
CONNECTOR = f"arn:aws:lambda:{REGION}:aws:network-connector:aws-network-connector"
WORKER_PORT = 8888
SESSION = "smoke-mvm-001"


def main() -> None:
    config.load_kube_config()
    dyn = dynamic.DynamicClient(api_client.ApiClient())
    d = Dispatcher(
        dyn,
        namespace=NS,
        microvm_execution_role_arn=EXEC_ROLE,
        microvm_ingress_connector_arns=[f"{CONNECTOR}:ALL_INGRESS"],
        microvm_egress_connector_arns=[f"{CONNECTOR}:INTERNET_EGRESS"],
    )

    tiers = d.tiers()
    print(f"[1] tiers: { {k: v.axis for k, v in tiers.items()} }")
    assert "microvm" in tiers, "MicrovmImage with agent-sandbox/tier label required"
    assert tiers["microvm"].axis == "off-cluster"

    t0 = time.monotonic()
    b = d.bind(SESSION, "microvm", timeout=300)
    bind_s = time.monotonic() - t0
    print(f"[2] MICROVM bind={bind_s:.2f}s endpoint={b.endpoint}")
    assert b.endpoint, "binding must surface status.endpoint"

    # Port-scoped auth token via the Lambda MicroVMs API — the proxy
    # requires X-aws-proxy-auth on every request.
    mvm_api = dyn.resources.get(
        api_version=f"{MICROVM_GROUP}/{MICROVM_VERSION}", kind="Microvm"
    )
    vm = mvm_api.get(name=b.name, namespace=NS)
    microvm_id = vm.status.microvmID
    lam = boto3.client("lambda-microvms", region_name=REGION)
    token = lam.create_microvm_auth_token(
        microvmIdentifier=microvm_id,
        expirationInMinutes=10,
        allowedPorts=[{"port": WORKER_PORT}],
    )["authToken"]
    headers = dict(token) if isinstance(token, dict) else {"X-aws-proxy-auth": token}
    # The proxy defaults to port 8080 inside the VM; route to the
    # worker's port explicitly (must be within the token's allowedPorts).
    headers["X-aws-proxy-port"] = str(WORKER_PORT)

    url = b.endpoint if b.endpoint.startswith("https://") else f"https://{b.endpoint}"
    req = urllib.request.Request(url + "/", headers=headers)
    body = json.loads(urllib.request.urlopen(req, timeout=30).read())
    print(f"[3] worker responded through proxy: {body}")
    assert body.get("status") == "ok" and body.get("tier") == "microvm"

    released = d.release(SESSION)
    print(f"[4] release: {released} resource(s)")
    assert released == 1
    assert d.release(SESSION) == 0, "release not idempotent"
    # CRs with a deletionTimestamp are ACK finalizer teardown in
    # flight (the VM is terminating) — not orphans.
    left = [
        i
        for i in (
            mvm_api.get(
                namespace=NS, label_selector="agent-sandbox/managed-by=dispatch"
            ).items
            or []
        )
        if not getattr(i.metadata, "deletionTimestamp", None)
    ]
    assert not left, f"orphaned Microvms: {[i.metadata.name for i in left]}"
    print("[4] released, idempotent, zero orphans")
    print(f"\nPASS: microvm tier E2E smoke (bind={bind_s:.2f}s)")


if __name__ == "__main__":
    main()
