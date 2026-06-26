---
sidebar_label: Image distribution at fan-out scale
---

# Image distribution for agent sandboxes at fan-out scale

Guidance for high-churn workloads — agent fleets, reinforcement-learning
fan-out, CI matrices — that create large, rapidly-changing sets of task-specific
sandbox images and need them on nodes fast. This page is the *use-case lens*: it
composes the mechanisms documented in
[Accelerating the pull process](../2-accelerate-pull-process/index.md)
(SOCI/Nydus snapshotters, Bottlerocket data-volume prefetch) and applies them to
the agent-sandbox tiers ([Agent Sandbox on EKS](../../../infra/agents/agent-sandbox.md)),
rather than re-explaining them.

## The fan-out shape

High-churn agent and RL workloads pull images unlike a typical service: a run can
spin up a large number of **distinct** images — frequently one per task variant —
each backing a short-lived sandbox pod. Individual images aren't unusually large;
the **count and churn** are what stress the pull path. Fully pulling and unpacking
every image on cold nodes during a launch burst dominates time-to-first-useful-work.

The key property to exploit is that these images are **mostly shared**: they're
built on common base and runtime layers, and differ only in a comparatively small
task-specific top layer. So the distribution problem decomposes into **a small set
of shared base layers plus many small per-task deltas** — which drives the whole
strategy: **pre-warm the shared layers, stream the deltas.**

## Two halves

### Pre-warm the shared layers

Bake the common base + runtime layers into the node's container store once, so
only per-task deltas move at creation time. Use the Bottlerocket data-volume
snapshot pattern in
[Prefetching images](../2-accelerate-pull-process/14-prefecthing-images-on-br.md):
build a golden snapshot containing the shared layers and mount it on every node
via the `bottlerocket_data_disk_snapshot_id` input. One immutable, versioned
snapshot, identical across nodes — regenerated only when the (slow-moving) base
set changes, rather than per-node cache volumes to track.

### Stream the per-task delta

Let the [SOCI snapshotter](../2-accelerate-pull-process/12-containerd-snapshotter.md)
handle the unique task layer. SOCI is enabled by default
(`enable_soci_snapshotter = true`) and runs **parallel-pull-unpack** with tuned
concurrency on instance-store NVMe. For large task layers where the workload
touches only a fraction at startup, **lazy-load** streams content on demand from a
SOCI index so the container can reach `Ready` before the full image transfers.

Build task images so common dependency bundles are their own layers:
containerd/SOCI deduplicate by digest, so shared layers pre-warm with the bases
and the unique delta that streams shrinks. Generate SOCI indexes in your image
build (automated via the
[ECR SOCI Index Builder](https://github.com/awslabs/cfn-ecr-aws-soci-index-builder),
or scripted) — build on SOCI v2.

## What's supported where (read this before choosing a pull mode)

This distinction is worth being precise about:

| Pull mode | What it does | AL2023 | Bottlerocket |
|---|---|---|---|
| **parallel-pull-unpack** (default) | Concurrent download + unpack of the full image. No SOCI index required; helps every image. | ✅ | ✅ |
| **lazy-load** | On-demand streaming from a SOCI index — fast start when only part of the image is read at launch. Un-indexed images fall back to parallel-pull. | ✅ (see gates below) | ❌ — Bottlerocket's settings API exposes only `parallel-pull-unpack` |

Lazy-load and parallel-pull are **not** combined per image — it's either/or.
With the fallback enabled (our config sets `experimental_parallel_pull_as_fallback`),
an image **with** a SOCI index lazy-loads, while an image **without** one falls
back to parallel-pull (never the slow sequential default). Lazy-load's per-layer
fetch is actually *slower* than parallel-pull, so its payoff is **startup
latency** on large, sparsely-accessed images — not full-image throughput. If a
workload reads most of the image at start, parallel-pull is the better choice.
Parallel-pull-unpack needs no index and helps every image; choose lazy-load
(AL2023 only) when images are SOCI-indexed and the startup working set is sparse.
On Bottlerocket, parallel is the only supported mode today.

## Enabling lazy-load on AL2023 — two gates that silently disable it

Both must be satisfied, and each fails *silently* (soci looks installed/active, but no lazy-load happens). Validated on a live cluster, June 2026:

1. **Instance size — xlarge+.** `nodeadm` only wires soci into containerd (proxy plugins + `snapshotter = soci` + kubelet image-service routing) when the node has **≥4 vCPU and ≥7 GiB** (the `UseSOCISnapshotter` gate). On a 2-vCPU node, soci is never in the pull path — no error, just a full pull. Require xlarge+ on the pool (`karpenter.k8s.aws/instance-cpu Gt "3"`).
2. **Explicit pull-modes config — soci's default is NOT lazy.** A minimal `cri_keychain`-only config does **not** lazy-load. You must explicitly enable `[pull_modes.soci_v1]` and `[pull_modes.soci_v2]` (with `experimental_parallel_pull_as_fallback` for un-indexed images). This was verified empirically: same node, same soci version — `cri_keychain`-only gave **0** FUSE mounts; the explicit config gave **19**.

Two ways to deliver the explicit config, both requiring xlarge+:

- **Today, no custom AMI (bridge)**: keep the EKS-managed `FastImagePull` feature gate (it installs + wires soci) and override `/etc/soci-snapshotter-grpc/config.toml` in userData with the explicit lazy config. nodeadm config generation runs before cloud-init, so the override wins. This is what the [agent-sandbox SOCI lazy nodepool](https://github.com/awslabs/ai-on-eks/blob/main/infra/agent-sandbox/nodepools/agent-sandbox-soci-lazy.yaml) ships.
- **Durable (cleaner)**: a `SociLazyLoading` nodeadm feature gate (proposed to `amazon-eks-ami`) that writes the explicit lazy config + wiring directly — no override hack. Same end state.



Image distribution is a node-bootstrap concern, independent of the sandbox
isolation runtime. The same SOCI path serves a `gvisor` sandbox and a `kata-fc`
sandbox alike — pick the [isolation tier](../../../infra/agents/agent-sandbox.md)
for your threat model and the pull mode for your image profile, independently.

## Scaling thresholds — where these techniques start to matter

Customers consistently ask "where do the limits actually land?" A blend of
published EKS scalability guidance and generalized field experience running
high-churn fleets:

- **Cluster scale triggers** (published): plan deliberately beyond roughly
  **300 nodes / 5,000 pods**, and treat any cluster creating/destroying
  **hundreds of resources per minute** as entering the regime where *churn* —
  not steady-state size — drives the scalability limits
  ([Data on EKS scalability](https://awslabs.github.io/data-on-eks/docs/bestpractices/scalability/),
  [EKS scalability best practices](https://docs.aws.amazon.com/eks/latest/best-practices/scalability.html)).
  The opt-in ultra-scale cluster capability is validated to ~**100K nodes**
  ([EKS ultra-scale](https://aws.amazon.com/blogs/containers/under-the-hood-amazon-eks-ultra-scale-clusters/)).
- **Scheduler throughput** (published): the default scheduler sustains on the
  order of **hundreds of pods/second** even at very large node counts, but it's
  serial. If your target sandbox-creation rate approaches that order, image
  distribution won't move it — you need scheduler sharding, batch/gang
  admission, or a lifecycle that doesn't route every sandbox through per-pod
  scheduling.
- **Image-distribution onset** (field-generalized): the pull path becomes the
  dominant cold-start cost once a run spans **thousands or more distinct images**
  and nodes are materializing **dozens of distinct task layers concurrently**.
  Below that, default parallel-pull-unpack is usually enough; above it, pre-warm
  the shared layers and lazy-load the deltas.
- **Pod density under churn** (field-generalized): high-churn sandbox density
  commonly lands **well below a node's theoretical max — often in the tens of
  pods per node** — because per-pod networking (IP/ENI allocation) and runtime
  overhead bind before CPU/memory do. Treat a low observed ceiling as a signal
  to investigate the data plane (VPC CNI prefix delegation / NAU budget, runtime
  choice), not just to add nodes.

These are directional starting points, not contractual limits — the right move
is always to validate against your own workload (below). And image distribution
is necessary but not sufficient at the top end: sandbox creation rate, pod
density, and session-state handling are co-equal concerns. Image distribution
keeps node-ready→pod-ready fast; it doesn't raise the scheduler's pods/sec
ceiling. Treat them as one pipeline.

## Validate it

Whether lazy-load actually engages for an indexed image on a given node
configuration is covered by a conformance spec
(`blueprints/agent-sandbox/conformance/soci-image-distribution/`) run through the
[conformance framework](https://github.com/awslabs/ai-on-eks/tree/main/infra/base/conformance).
Run it end-to-end (spin-up → conformance → cleanup) on a live cluster before
relying on a given pull mode.
