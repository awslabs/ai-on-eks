---
name: nvidia-dynamo-modelexpress-startup-timing
description: Analyze Dynamo/vLLM worker startup logs that use ModelExpress for weight loading (P2P/RDMA receiver, S3/model_streamer source, or artifacts) and produce a timing breakdown - weight load time, RDMA download time and speed (per-rank and node-aggregate across TP ranks), compilation cost, and total time-to-ready. Handles both source workers (S3 fallback + P2P-source publish) and RDMA receivers. Supports both a quick four-answer summary and a detailed phase-by-phase walkthrough of every startup step. Use when given a vLLM/Dynamo/ModelExpress worker log and asked to break down startup timing, download speed, or where startup time is spent.
version: 2.2.0
tags: [skill, modelexpress, dynamo, vllm, rdma, p2p, startup, timing, log-analysis, inference]
---

# ModelExpress vLLM Worker Startup Timing Analysis

## Overview

Dynamo vLLM workers that use ModelExpress (`load_format: modelexpress` / `mx`) emit
structured `[TIMING]` log lines during startup. This skill extracts them and produces either:

- a **quick summary** answering: full weight-load time, actual download (RDMA/P2P) time and
  speed, engine compilation cost, and total time-to-ready; or
- a **detailed six-phase walkthrough** attributing every second from process start to READY.

Works for P2P/RDMA transfers, and adapts to S3 or local artifact loads (same loader,
different strategy line).

## The six startup phases

Every run decomposes into these phases. Anchor each with the marker in the next section and
compute durations from line timestamps.

1. **Runtime boot & Dynamo registration** - process start, CuPy/CUDA context, NVFP4/Quack/
   inductor patches, MxModelLoader registration, K8s discovery, system status server.
2. **EngineCore import & config resolution** - the EngineCore child process imports
   torch/vLLM (a large, mostly silent gap), then resolves model config and `Initializing a
   V1 LLM engine`. Architecture resolution happens here (see note below).
3. **Worker fork + NCCL / distributed init** - TP workers fork and re-import, `parallel_state`
   NCCL init, all-reduce backend selection, rank assignment.
4. **Weight load (MxModelLoader)** - artifact-cache fetch, backend/kernel selection, MoE prep,
   RDMA tensor registration, P2P metadata fetch, the actual transfer, loader finalize.
5. **Engine init: compile + warmup + KV + autotune + capture** - the single
   `init engine (profile, create kv cache, warmup model) took N s (compilation: M s)` line
   covers this whole phase; its sub-steps are torch.compile (AOT load or cold compile),
   initial profiling/warmup run, KV cache creation, kernel warmup + FlashInfer autotune, and
   CUDA graph capture.
6. **Registration & readiness** - Dynamo endpoint registration (`generate` etc.), metadata
   publish, `Status -> READY`.

## Log markers (source of truth)

Match these substrings. ModelExpress timing lines contain `[TIMING]`. Every line is prefixed
with a timestamp (`YYYY-MM-DDThh:mm:ss...Z`); use it for deltas. Report rank-0 / EngineCore
lines; other ranks track within ~1 s.

| Phase | Grep marker | What it reports |
|---|---|---|
| 1 | first line / `dynamo_runtime::worker` | t0 for total time |
| 1 | `Utilizing CuPy` | CUDA/CuPy context init (dominant early cost) |
| 1 | `NaN-clamp`, `Quack FP8`, `inductor factorable` | patch install (sub-second each) |
| 2 | `Resolved architecture:` | model class chosen (dictates attention/MoE/KV backends) |
| 2 | `Using max model len`, `_validate_cache_dtype`, `Chunked prefill`, `log_enabled_passes` | config-resolution cluster (negligible cost, decision points) |
| 2 | `Initializing a V1 LLM engine` | config resolved; confirms model, TP, `load_format`, `quantization`, `cache_dir` (empty = cold compile) |
| 2/3 | `DP group leader` | executor spins up workers |
| 3 | `parallel_state.py:1638` | per-rank NCCL init begins |
| 3 | `pynccl`, `cuda_communicator` all-reduce backends | NCCL version + TP/EP backend selection |
| 3 | `parallel_state.py:1982` (`rank 0 ... assigned`) | rank/TP/EP assignment done |
| 4 | `Starting to load model` | start of weight load |
| 4 | `MxModelLoader starting` | confirms `p2p_enabled` |
| 4 | `Artifact install complete` / `vLLM artifact install complete` | artifact cache fetched (name, MiB, elapsed) - e.g. torch_compile_cache |
| 4 | `No ready vLLM artifact source for` | that cache had no source and is rebuilt locally (extra cost later) |
| 4 | `Selected .* Kernel`, `Using .* attention backend`, `MLA prefill`, `.* MoE backend`, `MoEPrepareAndFinalize` | backend/kernel selection + MoE plumbing (can be a quiet multi-second window) |
| 4 | `Eligible loaders:` / `Trying strategy:` | rdma vs model_streamer vs default (source workers fall through `rdma` to `model_streamer`) |
| 4 | `Streaming weights from s3://` / `Model streamer weight loading complete` | S3 model_streamer download window (bracket these two for S3 download time) |
| 4 | `Registered .* non-contiguous tensors`, `Registering tensors: .* GB` | tensor prep before RDMA registration |
| 4 | `[TIMING] register_tensors:` | RDMA region registration (usually dominant part of receive) + total GB |
| 4 | `[TIMING] P2P NIXL metadata fetch:` | remote agent metadata |
| 4 | `[TIMING] RDMA transfer complete:` | **bytes on the wire**: tensors, GB, seconds, Gbps |
| 4 | `[TIMING] Total receive time:` | **end-to-end receive** = register + metadata + match/prep + transfer |
| 4 | `MxModelLoader.load_model() COMPLETE in` | full loader wall time |
| 4 | `Model loading took` | vLLM's own "GiB and seconds" for weights |
| 5 | `torch.compile took` / `Directly load AOT compilation` / `reconstructed serializable fn` | compile step; AOT-load = artifact cache hit (fast), otherwise cold compile |
| 5 | `Initial profiling/warmup run took` | first forward pass to size memory |
| 5 | `GPU KV cache size:` | KV cache created (+ max concurrency) |
| 5 | `Warming up .* router GEMM kernels` | MoE router GEMM warmup (`kernel_warmup.py`), few seconds; shape shows expert count |
| 5 | `DeepGEMM warmup: .*it/s` (tqdm bar) | FP8 block-scaled GEMM kernel warmup - can be the single largest sub-step. See note below on reading the tqdm bar. |
| 5 | `Using FlashInfer autotune cache file:` | path of the FlashInfer autotune cache (empty on cold start) |
| 5 | `[Autotuner]: Autotuning process starts/ends`, `Saved N configs to .../flashinfer_autotune_cache` | FlashInfer autotune window; `N new, 0 from previous` = cold (nothing cached) |
| 5 | `shm_broadcast.py:801 No available shared memory broadcast block found in 60 seconds` | other TP ranks blocked on the shm barrier while rank 0 does a long warmup/compile - expected noise, marks a long silent window |
| 5 | `Capturing CUDA graphs` (progress bars) / `Graph capturing finished in` | CUDA graph capture (PIECEWISE + FULL) |
| 5 | `init engine (profile, create kv cache, warmup model) took` | **total engine-init seconds AND compilation seconds** (the key roll-up line) |
| 6 | `Registering endpoint: ... endpoint=generate` | worker can serve |
| 6 | `Status -> READY` | fully ready / published as P2P source |
| 6 | `Source publish attempt failed` | (any worker publishing itself as a P2P source, receivers included) the publish did NOT complete - worker is READY/serving but is not yet exposing itself for RDMA pulls. Report the reason (e.g. `tarred artifact source contains symlink: ...`) as an operational flag. |

## Procedure

1. Locate the log path (often outside the workspace, e.g. a OneDrive/Downloads path). Use
   terminal `grep -nE` rather than the workspace grep tool, which only searches workspace roots.

2. For a **quick summary**, two passes suffice:

   ```bash
   LOG="/path/to/worker.log"
   # Pass 1: load + transfer + receive + engine-init/compile
   grep -nE "Starting to load model|MxModelLoader.*starting|Trying strategy:|Streaming weights from s3://|Model streamer weight loading complete|\[TIMING\] register_tensors:|P2P NIXL metadata fetch:|RDMA transfer complete:|Total receive time:|MxModelLoader.load_model\(\) COMPLETE|Model loading took|Graph capturing finished|init engine \(profile" "$LOG"
   # Pass 2: endpoints + total-time anchors (+ source-publish failures)
   grep -nE "Initializing a V1 LLM engine|endpoint=generate|Status -> READY|Source publish attempt failed" "$LOG" | head -20
   head -1 "$LOG"   # process start t0
   ```

3. For a **detailed walkthrough**, additionally extract the phase-boundary and intra-phase
   markers, then read each inter-milestone window to fill silent gaps. Filter to rank 0 /
   EngineCore and drop noise:

   ```bash
   # Phase 1-3 boundaries (boot, import, NCCL)
   grep -nE "Utilizing CuPy|Resolved architecture|Initializing a V1 LLM engine|DP group leader|parallel_state.py:1638|pynccl|cuda_communicator|parallel_state.py:1982|Starting to load model" "$LOG" | grep -vE "tower_http"
   # Phase 4 internals
   grep -nE "MxModelLoader starting|artifact install complete|No ready vLLM artifact|attention backend|MLA prefill|MoE backend|MoEPrepare|non-contiguous|Registering tensors|register_tensors:|metadata fetch:|RDMA transfer complete|Total receive time|load_model\(\) COMPLETE|Model loading took" "$LOG" | grep -E "Worker_TP0|Worker 0\]"
   # Phase 5 internals
   grep -nE "torch.compile took|Directly load AOT|Initial profiling|GPU KV cache size|Warming up|Autotuning process (starts|ends)|Graph capturing finished|init engine \(profile" "$LOG" | grep -E "Worker_TP0|EngineCore"
   ```
   To pretty-print a window, strip the double timestamp / pid prefixes:
   `sed -E 's/^[0-9T:.-]+Z [0-9T:.-]+Z //; s/\(Worker_TP0 pid=[0-9]+\)//; s/\(EngineCore pid=[0-9]+\)//; s/ INFO 09-[0-9]+//'`

5. **Break down the kernel-warmup tail (KV-cache-created -> graph-capture-finished).** This
   window is usually the second-biggest chunk of Phase 5 after cold compile, and most of it is
   a *silent* gap: the dominant sub-step (DeepGEMM warmup) is a tqdm progress bar that is
   **not** prefixed with a `Worker_TP0`/`EngineCore` tag, so it is invisible to the rank-0
   filters above. Extract it explicitly and read the untagged bar directly:

   ```bash
   # Named warmup sub-steps + the untagged tqdm bars (router GEMM, DeepGEMM, CUDA graph capture)
   grep -nE "router GEMM kernels|DeepGEMM warmup|Autotuning process (starts|ends)|autotune cache file|Saved [0-9]+ configs|Capturing CUDA graphs|Graph capturing finished" "$LOG"
   # First and last frame of the DeepGEMM bar give its window; the bar's own [MM:SS] elapsed
   # counter (e.g. 3338/3338 [04:35<00:00]) confirms the duration:
   grep -nE "DeepGEMM warmup" "$LOG" | sed -n '1p;$p'
   ```
   Attribute the DeepGEMM window from the surrounding line timestamps (first bar frame ->
   `Autotuning process starts`) and cross-check against the tqdm elapsed counter. The `it/s` on
   the final bar line is tqdm's average, not a phase boundary.

6. **Break down the multi-rank RDMA transfer (receivers, TP > 1).** Every `RDMA transfer
   complete` line is **per rank**: that rank's own shard (GB), its own transfer seconds, and its
   own Gbps. Ranks pull their shards from the rank-matched source GPU in parallel, so the
   node-wide rate is NOT the per-rank rate and NOT a sum of times. Reconstruct each rank's
   window (start = line timestamp - transfer seconds; it matches that rank's `prep_xfer_dlist`
   line within a few ms) and compute the node aggregates:

   ```bash
   grep -E "RDMA transfer complete:" "$LOG" | awk '
   { split(substr($1,12,18),t,":"); end=t[1]*3600+t[2]*60+t[3]
     match($0,/\[Worker [0-9]+\]/); r=substr($0,RSTART+8,RLENGTH-9)
     match($0,/[0-9.]+ GB, [0-9.]+s, [0-9.]+ Gbps/); split(substr($0,RSTART,RLENGTH),f,/[ ,s]+/)
     gb=f[1]; d=f[3]; g=f[4]; st=end-d
     printf "TP%-2s start=%s end=%s  %.3f s  %6.1f Gbps  %5.1f GB/s\n", r, fmt(st), fmt(end), d, g, gb/d
     n++; tot+=gb; sum+=g; if(d>max)max=d
     if(n==1||st<fs)fs=st; if(n==1||st>ls)ls=st; if(n==1||end<fe)fe=end; if(n==1||end>le)le=end }
   function fmt(s){ return sprintf("%02d:%02d:%06.3f", int(s/3600), int(s%3600/60), s-int(s/60)*60) }
   END{ w=le-fs
     printf "ranks=%d total=%.2f GB  start-skew=%.3f s  all-ranks-overlap=%.3f s\n", n, tot, ls-fs, (fe>ls?fe-ls:0)
     printf "effective node rate = %.1f Gbps (%.1f GB/s) over %.3f s wall\n", tot*8/w, tot/w, w
     printf "no-skew rate        = %.1f Gbps (slowest rank %.3f s)\n", tot*8/max, max
     printf "sum of per-rank     = %.1f Gbps (upper bound)\n", sum }' | sort
   ```
   Also grab each rank's `register_tensors:` and `P2P NIXL metadata fetch:` seconds - their
   spread is what causes the start skew. Check for `Could not deduce average EFA device upstream
   link bandwidth ... default (all) rail selection` (NUMA-aware rail pinning off), a likely cause
   of uneven per-rank rates.

4. These logs have TP-size near-identical worker lines per metric. Report a representative
   value plus the per-worker range (e.g. "~49.2 s, range 49.05-49.71 s"). Do not sum
   per-worker RDMA times - the workers transfer in parallel.

## Output format

**Units.** Report every individual duration in **seconds** (e.g. `~275 s`, `210.10 s`). Report
only the **total** time-to-ready in **minutes with seconds in parentheses** (e.g.
`~13.6 min (813 s)`). Apply this to the phase table, the headline answers, and the warmup-tail
table: phase/sub-step durations stay in seconds; the total row is minutes-with-seconds. Do not
convert individual sub-steps to minutes even when they are large (a ~275 s step stays `~275 s`,
not `~4 min 35 s`).

Start with a phase table (phase, window, duration, % of total) covering the six phases above.

Then give the four headline answers:

1. **Weight load (full):** from `MxModelLoader.load_model() COMPLETE` / `Model loading took`.
2. **Download time and speed:** depends on the strategy that actually loaded weights:
   - **RDMA receiver** - distinguish the two RDMA layers:
     - **Raw transfer** from `RDMA transfer complete` (GB, seconds, Gbps) - bytes on the wire.
     - **Total receive** from `Total receive time` - end-to-end incl. RDMA registration.
       Note the gap = registration/metadata setup.
   - **S3 / model_streamer (typical of a source worker)** - there is no RDMA receive; report
     the S3 stream window (`Streaming weights from s3://` -> `Model streamer weight loading
     complete`) and state that no on-the-wire Gbps figure exists. The `register_tensors` line
     here is source-publish prep, not a download.
3. **Compilation:** the `(compilation: N s)` value inside `init engine ... took`. Small (~2 s)
   means an AOT/artifact cache hit (`Directly load AOT compilation`); large (100s+) means cold
   compile (engine config shows `cache_dir: ''` and no artifact source).
4. **Total time to ready:** process-start timestamp -> `endpoint=generate` (servable) and ->
   `Status -> READY` (fully ready). Report both.

For a detailed request, add a per-phase step table with timestamps and deltas, then a
finest-grain cost ranking (which individual sub-steps cost the most). When Phase 5 is a large
share of startup, also break down the **kernel-warmup tail** (KV-cache-created ->
graph-capture-finished) into its sub-steps as its own table: router GEMM warmup, DeepGEMM
warmup, FlashInfer autotune, and CUDA graph capture, with each window/duration. This tail
hides inside the single `init engine ... took` roll-up, so it needs to be reconstructed from
the markers above (see procedure step 5).

**Multi-rank transfer breakdown (always include for RDMA receivers with TP > 1, quick or
detailed).** In headline answer 2, label the raw-transfer figure explicitly as **per rank**
(e.g. "42.11 GB per rank in 1.69 s @ 199.1 Gbps on rank 0; range 199-318 Gbps") and give the
effective node rate next to it. Then add a section produced by procedure step 6:

1. A one-line mechanism note: N shards (one per GPU), each rank pulls its own shard from the
   rank-matched source GPU, N parallel RDMA streams; per-rank rate = shard / rank time; node
   rate = total bytes / wall window.
2. A per-rank table sorted by start time: rank, start, end, time (s), Gbps, GB/s. Under it:
   per-rank range + mean, start skew (first start -> last start), and the all-ranks-overlap
   window (last start -> first end).
3. A node-aggregate table with three rows and how each is computed:
   - **Effective node rate** (the real number): total GB / (first start -> last end).
   - **No-skew rate**: total GB / slowest rank's time (what simultaneous starts would give).
   - **Upper bound**: sum of per-rank Gbps (only if every rank ran fully overlapped).
4. Why effective < upper bound: start skew from per-rank `register_tensors` / metadata-fetch
   spread, plus slow-rank outliers (name them; note rail-selection warnings if present).
5. How to read it: per-GPU wire speed = per-rank rate; whole-model node speed = effective node
   rate; never sum per-rank times; slowest rank sets completion; full download step = per-rank
   `Total receive time`.

Close with the takeaway: which phase dominates, and the highest-leverage next optimization.

## Interpretation notes

- **Raw transfer << total receive << loader total.** Examples seen in practice: raw transfer
  ~1 s @ ~772-774 Gbps; total receive ~7.5-8 s (registration ~4.5-5.3 s dominates); loader
  total ~47-49 s (backend selection, MoE prep, tensor prep, serialization around the receive).
- **~95.7 GB per worker** in `register_tensors`/`RDMA transfer complete` is the per-GPU shard;
  multiply by TP size for the aggregate moved across the node, but keep per-worker speed as-is.
- **Per-rank vs node transfer rate.** Observed (GLM-5.3-Flash TP8, g7e.48xlarge, 42.11 GB/rank):
  per-rank 199-318 Gbps; start skew 0.72 s and only ~0.42 s with all 8 ranks overlapped;
  effective node rate ~1.18 Tbps (336.9 GB / 2.28 s wall) vs ~1.59 Tbps no-skew and ~2.08 Tbps
  sum-of-ranks upper bound. Skew from staggered `register_tensors` completion is the main gap -
  quote the effective node rate, not the sum.
- **Artifacts vs no-artifacts is the big lever.** Same model/weights, the divergence is entirely
  in Phase 5 compilation. Observed: with warm artifacts compilation ~1.9 s and total ~3.4 min;
  cold (no artifacts) compilation ~210 s and total ~12.3 min (~3.6x). If some caches show
  `No ready vLLM artifact source` (e.g. `tilelang_cache`, `cute_dsl_cache`, flashinfer autotune),
  those get rebuilt locally and inflate the warmup/autotune sub-steps - flag them as
  seed-able artifact opportunities.
- **The kernel-warmup tail is a second cold-start lever, separate from torch.compile.** After
  KV-cache creation, Phase 5 runs router GEMM warmup (seconds), then **DeepGEMM warmup**, then
  FlashInfer autotune, then CUDA graph capture. On a cold start with no seeded DeepGEMM cache,
  DeepGEMM warmup dominates this tail: observed ~4 min 35 s to JIT+benchmark 3338 FP8
  block-scaled kernels (~12 it/s) for a GLM-5.3 FP8 MoE (TP8) worker - about 84% of the
  warmup tail and ~34% of the whole engine-init phase, second only to the 210 s cold compile.
  Its cost tracks the number of GEMM shape/config variants, which follows from the FP8 kernel
  pick (`DeepGemmFp8BlockScaledMMKernel`). FlashInfer autotune (~29 s, `N new, 0 from previous`)
  and CUDA graph capture (~18 s) are the smaller remainder. Between cold `torch.compile` and
  cold DeepGEMM warmup, two uncached steps can be ~8 min of a ~13.5 min startup - seeding a
  **persistent DeepGEMM cache** alongside the torch.compile/inductor and FlashInfer autotune
  caches is the highest-leverage fix after the compile cache itself.
- **`Resolved architecture:` decodes the model shape.** e.g. `GlmMoeDsaForCausalLM` = GLM family
  + Mixture-of-Experts + DeepSeek Sparse Attention (DSA/MLA) + causal LM. It drives the later
  backend picks (FLASHINFER_MLA_SPARSE attn, TRTLLM_RAGGED prefill, FLASHINFER_TRTLLM FP8 MoE,
  DEEPSEEK_V32_INDEXER KV block size). It is a sub-second decision point, not a time sink.
- **Hidden framework cost.** Python/torch import in EngineCore (Phase 2, tens of seconds, mostly
  a silent gap) plus worker fork + NCCL init (Phase 3) can together be ~30-35% of startup and
  are easy to overlook because they precede any weight or GPU work.
- **`503 Service Unavailable` from `tower_http`** during startup is the health endpoint before
  readiness - expected noise, not an error to report.
- **Source worker vs receiver - read them differently.** Ignore the log filename and pod name
  when classifying; decide only from which strategy actually loaded the weights. A **receiver**
  logs `RDMA transfer complete` / `Total receive time` (plus `Source selection` / `Fetched N
  tensors from worker at <ip>`). A **source** is the *first* worker up, with no upstream P2P
  source to pull from yet, and logs `Streaming weights from s3://`. `Source publish` lines do
  not identify a source - receivers also publish themselves as sources once READY. A source
  worker's `Eligible loaders: ['rdma', 'model_streamer', ...]`
  tries `rdma` first, fails to find a source, and **falls back to `model_streamer` (S3)** -
  look for `model_streamer_strategy.py ... Streaming weights from s3://...` and
  `Model streamer weight loading complete`. On a source worker:
  - There is **no** `RDMA transfer complete` / `Total receive time`. The download is the S3
    stream: bracket it from `Streaming weights from s3://` to `Model streamer weight loading
    complete` (observed ~67 s for GLM-5.3 TP8). No on-the-wire Gbps figure exists for S3.
  - The `[TIMING] register_tensors:` line on a source is **source-publish prep** (registering
    the ~95.7 GB shard as NIXL regions so *downstream* workers can pull it), NOT a receive.
    Do not present it as download time.
  - Check the tail for `Source publish attempt failed` (see markers). The worker can reach
    `Status -> READY` and serve while still failing to publish itself as a P2P source - flag
    it, since downstream receivers then have nothing to RDMA from.
- For S3/artifact loads there is no `RDMA transfer complete`; use the S3 stream window (or
  `Model loading took`) for the download and still use the `init engine ... took` line for
  compilation.
