# Sandbox Dispatch — design

**Status**: v0 design, pre-implementation. Owner: Brian Hammons.

## The one-sentence design

The dispatcher is a thin session→execution-target binder that treats the
Kubernetes control plane as its registry and its state store — it never
reconciles anything, because the agent-sandbox controller (in-cluster
tiers) and the ACK lambdamicrovms controller (off-cluster tier) already
do.

## Anti-goals (learned, not asserted)

- **No rebuilt control plane.** The cautionary pattern is well
  documented: platforms that re-implement scheduling/lifecycle on top
  of an existing orchestrator accumulate opaque failure modes (instances
  "ready" before warm, cascading respawns). Everything stateful here is
  a CR owned by an existing controller; the dispatcher holds no state
  that `kubectl get` can't show.
- **No tool surface.** Tool execution (exec / files / status) belongs to
  the upstream agent-sandbox MCP server + SDK. The dispatcher only
  decides *where* a session runs and hands back a connection reference.
- **No config file registry.** Tiers are discovered from the cluster,
  not declared in dispatcher config (see below). Adding a tier is
  `kubectl apply`, not a dispatcher change.

## Label-based tier registry

The cluster is the registry. A **tier** is any execution capacity
carrying the `agent-sandbox/tier` label:

| Resource | Axis | Example |
|---|---|---|
| `SandboxWarmPool` labeled `agent-sandbox/tier: gvisor` | in-cluster (RuntimeClass ladder: runc / gvisor / kata-fc, extensible to future runtime classes) | pool → template → runtimeClass |
| `MicrovmImage` labeled `agent-sandbox/tier: microvm` | off-cluster (Lambda MicroVM via ACK) | image → per-session `Microvm` CR |

Discovery is a namespaced label list at bind time (cheap, no cache to
invalidate). The label value is the tier name callers ask for; which
provider handles it follows from the resource kind that carries it.
Multiple pools may carry the same tier label — the registry picks by
`agent-sandbox/priority` annotation, then name, making blue/green
template rollouts a pure labeling operation.

## Session binding

`bind(session_id, tier)`:
1. Registry resolves tier → (provider, capacity resource)
2. Provider creates the per-session CR, labeled
   `agent-sandbox/session-id: <id>` and `agent-sandbox/managed-by: dispatch`
   - in-cluster: `SandboxClaim` with `warmPoolRef` → warm checkout or
     cold start, controller's choice
   - off-cluster: `Microvm` with `imageIdentifierRef` → managed
     Firecracker with idle-suspend/auto-resume
3. Provider resolves the connection reference:
   - claim: `status.sandbox.name` (pod name == sandbox name is a v1.0
     guarantee; never assume claim name — warm-adopted sandboxes are
     pool-named)
   - microvm: `status.endpoint` once `status.state == Running`
4. Returns a `Binding` (session, tier, kind, name, endpoint/pod ref)

`release(session_id)`: delete the session's CRs by label selector;
controllers cascade teardown. Idempotent — releasing an unknown session
is a no-op. The operational surface is deliberately just labels:
`kubectl get sandboxclaims,microvms -l agent-sandbox/session-id=X`.

## Lifecycle hooks

Small, synchronous callbacks at `pre_bind / post_bind / pre_release /
post_release`, receiving the `Binding`. This is the extension seam for
everything that is *not* the dispatcher's job:

- CMA work-queue integration (dispatcher-as-CMA-worker polls the queue
  in a loop and calls bind/release per session)
- observability + attribution (per-user quota tracking, cost
  attribution from gateway down to sandbox — hooks stamp the labels the
  metrics pipeline aggregates on)
- admission policy (tier allow-lists per caller)

Hooks fail open by default (log, don't block) except `pre_bind`, which
may veto (that's the admission point).

## Worker runtime requirement

Templates behind dispatched pools must run an SDK-compatible runtime
(python-runtime-sandbox HTTP on :8080, or sandboxd) so the MCP
server/SDK can execute inside them. The sleep+exec template shape from
the earlier agent-with-tools blueprint is not dispatchable.

## Failure semantics

- Bind timeout: claim not bound / microvm not Running within deadline →
  provider deletes what it created (no orphaned CRs) and raises.
- Dispatcher crash: no in-memory state matters; sessions are
  reconstructable from labels. A restarted dispatcher can resolve or
  release any existing session.
- Tier absent: fail fast at registry lookup with the available-tier
  list in the error.

## Prior art and lineage

- **kubernetes-sigs/agent-sandbox#1267** (closed stale, unimplemented) —
  proposed isolation-tier routing for SandboxClaims via a mutating
  admission webhook + ConfigMap pool mapping. This design answers the
  same need out-of-band: a binder instead of a webhook (no admission
  latency or webhook failure modes on the claim path) and the cluster
  itself as the registry instead of a ConfigMap. The issue's benchmark
  data motivates warm pools per tier: **warm claim latency is
  runtime-independent (~0.32s across runc/gVisor/kata), while cold
  starts diverge (runc ~0.32s, gVisor ~1.1s, kata 8-13s)** — the
  stronger the isolation, the more the warm pool pays for itself. A
  commenter's on-demand→spot pool-fallback ask maps directly to the
  `agent-sandbox/priority` mechanism here.
- **Upstream `kueue-agent-sandbox` example** — Kueue as admission
  control (quota/queueing) for sandbox workloads. Complementary:
  `pre_bind` should delegate quota decisions to Kueue rather than grow
  its own; the hook stays a veto point, not a quota engine.
- **Upstream `hermes-agents-as-a-service` example** — the per-user
  platform shape (warm pool + tiny claim per user + suspend/wake).
  Validates session→claim binding; single-tier, no routing layer.
- **Karpenter** — the same philosophy one layer down: label/requirement-
  driven selection over declared capacity, cluster as registry. This
  dispatcher is that idea at session level.
- **Virtual Kubelet providers** (Fargate/ACI) — kin of the Microvm
  provider, but VK rebuilds a node abstraction; here off-cluster
  capacity stays a CR owned by its own controller (ACK).
- **kro** — complementary shape-holder: an RGD can compose each tier's
  bundle (template + pool + labels + egress policy) declaratively while
  dispatch remains the runtime binding.

No public implementation found that dispatches across both axes
(in-cluster RuntimeClass tiers + managed-service CRs) behind one label
registry; #1267 is the community-demand receipt for the in-cluster half.
