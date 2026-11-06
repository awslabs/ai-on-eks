---
sidebar_label: FSx for Lustre over EFA
---

# FSx for Lustre over EFA on EKS

## Overview

Loading very large model weights (hundreds of GB) from shared storage at pod start is the dominant
cost of cold-starting large-model inference — the service is unavailable until the weights are in GPU
memory. How fast they load depends on the storage **transport**:

- **TCP** is the standard network path — reliable, but bounded by the EC2 single-flow network limit and
  per-packet processing overhead.
- **EFA (Elastic Fabric Adapter)** uses RDMA to bypass the kernel network stack, so parallel reads from
  FSx for Lustre scale to multiple GB/s and model-load time drops substantially.

This guide covers running FSx for Lustre with EFA on EKS: when it helps, the one hard networking
requirement, the setup (enable → place nodes → verify), how to tune it, and the cost trade-offs.

> For an end-to-end workload using this pattern, see the
> [multi-node Llama-3.1-405B blueprint](../blueprints/inference/framework-guides/GPUs/vllm-llama3.1-405b-lws-gpu-fsx.md).

## When EFA helps — and when it doesn't

EFA's benefit scales with **read parallelism**, and it only accelerates the stages that are actually
storage-bound:

- **Large benefit — many-threaded, storage-bound reads.** Preloading or streaming weights with many
  parallel streams is where EFA shines: throughput scales with concurrency instead of hitting the TCP
  floor.
- **Modest benefit — once the bottleneck leaves storage.** EFA makes the *read* from FSx faster, but it
  can't speed up what happens next — moving the weights onto the GPUs (the CPU→GPU PCIe bus) and
  splitting the model across them (GPU tensor distribution and NCCL). When that GPU-side work is the
  limit, EFA has little left to give.
- **No benefit — single-threaded reads.** With no parallelism to amortize RDMA setup, plain TCP is
  marginally faster. Don't enable EFA for single-threaded access patterns.

### Why the non-EFA path plateaus — an EC2 network floor, not "slow TCP"

The ceiling on the non-EFA path is best understood as a **single-flow network limit on the EC2
instance** — not a limit of the TCP protocol or of FSx for Lustre. Lustre-over-TCP rides on a small,
bounded number of connections, so its read traffic stays within the per-flow regime and **adding reader
threads does not push it higher** — the client is out of network-flow headroom, not storage or CPU. You
cannot tune this away at the application layer (more threads, bigger blocks, and Lustre client tunings
do not add network flows). The documented ways to raise a single flow are infrastructure-level:

| Mechanism | Max single-flow bandwidth |
|-----------|---------------------------|
| Default (no placement group) | 5 Gbps |
| Cluster placement group | up to 10 Gbps |
| ENA Express (same-AZ, eligible instances) | up to 25 Gbps |
| Multipath TCP (MPTCP) | spreads one transfer across multiple paths |

None of these match EFA. **EFA removes the ceiling entirely** — RDMA bypasses the kernel TCP stack and
the single-flow cap, so throughput scales with concurrency up to the filesystem's provisioned limit.
(Single-flow limits per the
[EC2 instance network bandwidth documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-network-bandwidth.html).)

## Prerequisite: same AZ **and** same /16 CIDR

EFA is a single-AZ fabric and does not cross Availability Zones. For FSx-Lustre-over-EFA to engage, the
**client (GPU) nodes and the FSx filesystem must be in the same Availability Zone _and_ the same /16 VPC
CIDR block.**

- Different **AZ** than FSx → no EFA path at all.
- Different **/16** (for example an EKS secondary CIDR such as `100.64.0.0/16`, commonly used for pod
  density) → the EFA data path does **not** engage. The client's `@efa` interface still comes up and FSx
  even advertises an `@efa` NID, but it is unreachable across the /16: the mount still succeeds over
  `@tcp`, while the first large read stalls rather than cleanly falling back. Either way you lose the EFA
  speedup. This is the most common and most confusing failure mode.
- **Same /16, different subnet is fine.** The requirement is the /16, not FSx's exact subnet — a client
  in another subnet within the same /16 engages `@efa` normally.

**Sizing:** give the FSx-co-located subnet enough room for the EFA network interfaces — a p5.48xlarge
exposes many EFA interfaces per node, so a `/23` (or larger) subnet is a good default to avoid IP
exhaustion during scale-out.

## Step 1 — Enable EFA on the FSx filesystem

Set the FSx-EFA options in your blueprint's `blueprint.tfvars` (see the infra wrapper), then deploy the
base:

```hcl
deploy_fsx_volume                = true
fsx_efa_enabled                  = true   # EFA (RDMA) mode on the FSx filesystem
enable_aws_fsx_csi_driver        = true   # mounts the static fsx-static-pvc into pods
enable_aws_efa_k8s_device_plugin = true   # advertises vpc.amazonaws.com/efa to pods
# fsx_storage_capacity           = 19200  # optional — see "Tuning" for stripe/OST behavior
```

With `fsx_efa_enabled = true` the base provisions the filesystem as `PERSISTENT_2` with
`per_unit_storage_throughput = 1000`, a default `storage_capacity` of `4800` GiB, an `AUTOMATIC`
metadata configuration, and the self-referencing EFA security-group rules.

> **EFA mode is create-only (immutable).** You cannot toggle EFA on an existing FSx filesystem —
> changing `fsx_efa_enabled` replaces the filesystem. Decide before first apply.

## Step 2 — Place the GPU nodes (three capacity methods)

Whatever provisions the GPU nodes must put them in the FSx subnet (same AZ + /16) **and** install the
FSx Lustre client with EFA (LNet/RDMA). The stock AL2023 EKS AMI ships neither the Lustre client nor the
EFA LNet config, so node bootstrap must add them.

### Method A — Karpenter (default, shipped)

When `deploy_fsx_volume` and `fsx_efa_enabled` are both true, the existing EFA P-series GPU NodePool
(one per EFA instance type) uses a FSx-pinned `EC2NodeClass`: its `subnetSelectorTerms` is pinned to the
FSx filesystem's subnet — derived directly from the filesystem — and a same-AZ topology requirement is
applied, so co-location (same AZ + same /16) is guaranteed. The same NodeClass `userData` installs the
Lustre client + EFA config. This is the recommended path — no extra work.

### Method B — static managed node group with an Open ODCR

Create the node group **in the FSx subnet** with EFA enabled, and consume an **Open On-Demand Capacity
Reservation** via the node group's capacity-reservation settings. The reservation must match the
instance type + AZ and be `active` before you apply.

### Method C — static managed node group with a Capacity Block / targeted ODCR

Same as Method B, but targeting a **Capacity Block** or a targeted ODCR (common for scarce P-series
capacity). Pin the node group to the reservation and to the FSx subnet/AZ.

> **AL2023 bootstrap gotcha (static node groups):** on an AL2023 GPU node group
> (`ami_type = AL2023_x86_64_NVIDIA`), `pre_bootstrap_user_data` is **silently ignored** — AL2023 uses
> **nodeadm**, so custom pre-bootstrap (installing the Lustre client + EFA config) MUST go through
> **`cloudinit_pre_nodeadm`**. Getting this wrong brings the node up TCP-only. (`cloudinit_pre_nodeadm`
> requires a custom launch template; `enable_efa_support = true` forces one, satisfying that.) Karpenter's
> equivalent is the EC2NodeClass `userData`, handled for you in Method A.

## Step 3 — Verify EFA is engaged

After the FSx mount, confirm traffic is actually on the EFA fabric (not silently on TCP). On a node that
has loaded weights, the EFA hardware counters should climb during the load:

```bash
# EFA hardware receive/transmit counters — should increase as weights load over EFA
cat /sys/class/infiniband/*/ports/*/counters/port_rcv_data
cat /sys/class/infiniband/*/ports/*/counters/port_xmit_data
```

If the counters stay flat while the model loads, the read is on TCP — re-check the same-AZ + same-/16
prerequisite (the node is almost certainly on the wrong subnet/CIDR).

## Tuning: stripe layout, capacity, and client parameters

**The storage lever is stripe layout, not the filesystem's OST count.** More capacity gives the
filesystem more Object Storage Targets (OSTs), but that does nothing on its own — what matters is the
**stripe count** (`lfs setstripe -c <N>`): how many of those OSTs each file is actually spread across. A
wide filesystem whose files are single-striped reads at single-OST speed.

1. **Stripe the model directory across OSTs** (`lfs setstripe -c <N>` *before* writing the weights).
   This is the lever that raises parallel read throughput.
2. **To add OSTs, scale FSx capacity — then restripe.** Growing a filesystem **does not restripe
   existing data**; only data written *after* the grow, with a wider stripe, benefits. Plan the stripe
   width up front, or re-write the weights after growing.
3. **Lustre client tuning** (`max_rpcs_in_flight`, `read_ahead`, checksums) gives **no meaningful gain
   when the load is funnel-bound** — don't reach for it first.

> **Necessary but not sufficient.** EFA plus striping raise the *storage* read ceiling, but end-to-end
> model-load time is often bound by the **weight-streamer → GPU PCIe path** (and, on the pod, hugepages)
> rather than storage. Once reads are striped and on EFA, the remaining lever is workload-side — see the
> [blueprint](../blueprints/inference/framework-guides/GPUs/vllm-llama3.1-405b-lws-gpu-fsx.md) for the
> pod-level settings (streamer concurrency, hugepages).

## Cost & operational considerations

- EFA mode uses the `PERSISTENT_2` 1000 MB/s/TiB throughput tier — higher throughput (and cost) than a
  basic FSx volume. Size the filesystem to your model footprint and the stripe width you need.
- The payoff is **time-to-ready**: shorter GPU-idle time while weights load. On expensive GPU nodes that
  idle time is usually the larger cost lever than the FSx line item.
- Loading **directly from FSx over EFA can replace the NVMe-preload workaround** — no init-container copy
  and nothing to re-copy on pod restart (instance-store NVMe does not survive a stop/start). That removes
  an operational step as well as a cost.
- `efa_enabled` and the throughput tier are **create-only**, so plan capacity (and stripe width) before
  first apply.
