# Sandbox Dispatch — quickstart

Session→execution-target binding across two axes behind one interface:
**in-cluster** isolation tiers (runc / gvisor / kata-fc via
SandboxWarmPools) and **off-cluster** AWS Lambda MicroVMs (via the ACK
`lambdamicrovms` controller). The Kubernetes control plane is the
registry and the state store — the dispatcher never reconciles anything.
Architecture and rationale: [DESIGN.md](DESIGN.md).

```
            ┌─ in-cluster:  SandboxClaim ⇦ SandboxWarmPool (tier label)
bind(session, tier) ──┤
            └─ off-cluster: Microvm CR ⇦ MicrovmImage (tier label)
```

## Prerequisites

- `infra/agent-sandbox` provisioned (platform manifests applied — see its README)
- For the `microvm` tier only: `enable_ack_lambdamicrovms = true` in
  `blueprint.tfvars` (installs the ACK controller + IRSA), plus the
  [microvm worker image](../microvm/README.md)
- Python 3.11+, `pip install -r dispatcher/requirements.txt`, a kubeconfig

## 1. Register tiers (apply labeled capacity)

```bash
kubectl apply -f manifests/worker-runc.yaml -f manifests/worker-gvisor.yaml
# Standard EKS with the kata-fc RuntimeClass installed:
kubectl apply -f manifests/worker-kata-fc.yaml
```

Each file ships a SandboxTemplate on the official
`python-runtime-sandbox` image (so the agent-sandbox SDK / MCP server
can execute inside dispatched workers) plus a SandboxWarmPool carrying
the `agent-sandbox/tier` label — **the label on the pool is the
registry entry**. Adding a tier is `kubectl apply`; no dispatcher
config, no restart. `replicas: 1` pre-warms one worker per tier
(measured: warm bind ~0.8s regardless of runtime; cold ~4× that on
gvisor, more on kata) — set `0` to trade latency back for zero idle cost.

## 2. Bind and release sessions

```python
from kubernetes import config, dynamic
from kubernetes.client import api_client
from dispatcher import Dispatcher

config.load_kube_config()   # or load_incluster_config()
d = Dispatcher(dynamic.DynamicClient(api_client.ApiClient()),
               namespace="agent-sandboxes")

print(d.tiers())                      # discovered from labels

b = d.bind("session-042", "gvisor")   # warm checkout or cold start
print(b.pod)                          # connect via the SDK / MCP server

d.release("session-042")              # deletes by label; idempotent
```

The connection reference comes from CR status — `status.sandbox.name`
for claims (never the claim name: warm-adopted sandboxes are
pool-named), `status.endpoint` for MicroVMs. Everything the dispatcher
creates carries the session label, so the operational surface is plain
kubectl:

```bash
kubectl get sandboxclaims,microvms -l agent-sandbox/session-id=session-042
```

## 3. Lifecycle hooks (optional)

```python
from dispatcher import PreBindVeto

def quota_gate(binding):
    if binding.tier.name == "kata-fc" and not approved(binding.session_id):
        raise PreBindVeto("kata-fc requires approval")

d.on("pre_bind", quota_gate)     # admission point — may veto
d.on("post_bind", stamp_attribution_labels)   # fail-open
```

`pre_bind` runs before any CR is created and may veto; the other hooks
(`post_bind`, `pre_release`, `post_release`) fail open. This is the
seam for CMA work-queue integration, per-user attribution, and quota —
see DESIGN.md.

## Off-cluster tier (Lambda MicroVM)

Label a `MicrovmImage` with `agent-sandbox/tier: microvm` and construct
the dispatcher with `microvm_execution_role_arn=...`. Binds create a
per-session `Microvm` CR (idle-suspend + auto-resume) and return
`status.endpoint`. Constraints worth knowing: ARM64-only, connector-based
egress (no FQDN allowlists — the in-cluster tiers keep the finer egress
story), 8h session ceiling. See [../microvm/](../microvm/README.md).

## Measured (fresh Standard EKS rig)

| Operation | Time |
|---|---|
| Warm bind, gvisor | 0.82s (+2.7s pod-ready) |
| Warm bind, runc | 0.80s (+2.2s pod-ready) |
| Cold bind, gvisor (pool exhausted) | 3.01s (+2.2s pod-ready) |

Warm binding is runtime-independent — consistent with the benchmarks in
[kubernetes-sigs/agent-sandbox#1267](https://github.com/kubernetes-sigs/agent-sandbox/issues/1267):
the stronger the isolation tier, the more the warm pool pays for itself.
