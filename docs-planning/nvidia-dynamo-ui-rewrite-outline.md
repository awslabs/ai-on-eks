# NVIDIA Dynamo on Amazon EKS: UI rewrite outline

Target file: `website/docs/blueprints/inference/framework-guides/GPUs/nvidia-dynamo.md`
Fork: https://github.com/malinich1/ai-on-eks/
Sources:
- Current page: https://awslabs.github.io/ai-on-eks/docs/blueprints/inference/framework-guides/GPUs/nvidia-dynamo
- Erez's blueprint commit: https://github.com/erezzarum/ai-on-eks/blob/ad263f7bce1363d10da6255989353c1dad7e2b23/blueprints/inference/nvidia-dynamo/README.md (plus the recipe READMEs, `model-cache/README.md`, the manifests, `blueprint.tfvars`, and `.agents/skills/nvidia-dynamo-modelexpress-startup-timing/SKILL.md`)
- Dynamo docs: https://docs.nvidia.com/dynamo/dev/knowledge-base/overview

UI components already in the site: `StatusGrid`, `SectionDivider`, `ProgressSteps`, `Tabs`/`TabItem`, `Admonition`, `CodeBlock` (with `raw-loader` for manifests), mermaid, `<details>`. A `SectionDivider` with an icon separates each top-level section.

Legend: **[NEW]** = not on the current page. **[ERZ]** = sourced from Erez's commit.

---

## Page header

- Warning admonition about GPU/Neuron access and Karpenter (keep).
- Collapsed **TL;DR** card:
  - One-paragraph definition of Dynamo
  - "What you get": 5 bullets
  - `StatusGrid`: EKS version, region, namespace, ArgoCD, Karpenter, EFA + NIXL, S3/EFS, first deploy 20–30 min
  - **[NEW]** "Jump to" row: Fit check · Deploy · Blueprints · Performance · Profiler
- Collapsed **GPU capacity** note: Capacity Blocks, plus the `ai-on-eks=true` tag.

---

## 1. Intro to Dynamo 🧠

- 1.1 What Dynamo is and isn't: an orchestration layer above engines (vLLM, SGLang, TRT-LLM). It doesn't replace the engine.
- 1.2 Why it matters at scale: running across multiple GPUs and nodes, KV-aware routing, prefill/decode split.
- 1.3 How Dynamo fits on AWS: EKS, EFA, Capacity Blocks. Link to the Dynamo docs.

UI: short prose, plus a tip `Admonition`: "New to Dynamo? Read sections 1–3, then jump to Blueprints."

---

## 2. Concepts and architecture 🏗️

- 2.1 Core building blocks (table): Frontend, Router, Worker, PrefillWorker/DecodeWorker, NIXL, ModelExpress, `DynamoGraphDeployment`.
- 2.2 Serving modes at a glance: aggregated vs disaggregated (keep the existing mermaid diagram).
- 2.3 Platform components, each with a one-line "why it's here": Dynamo operator, Grove, KAI scheduler, NVIDIA GPU Operator, EFA device plugin, EFS CSI, kube-prometheus-stack.
- 2.4 Reference architecture on EKS: Terraform → ArgoCD add-ons → `dynamo-system` (keep the existing mermaid diagram).
- 2.5 **[NEW]** Request lifecycle: mermaid sequence of client → frontend → router → prefill → KV transfer (NIXL/EFA) → decode.
- 2.6 **[NEW][ERZ]** EFA networking on EKS: AL2023 AMIs, `efa_network_interfaces_policy = "2"`, VPC CNI multi-NIC, prefix delegation, external SNAT.
  - UI: table of setting and why it matters, collapsed in `<details>`.
- 2.7 **[NEW][ERZ]** Three ways to load weights:
  - EFS PVC (`h100-agg`, `h100-disagg`)
  - Stream from S3 through ModelExpress (`*-mx-s3`)
  - S3 to local NVMe: an initContainer runs `aws s3 sync` with the CRT client to a hostPath under `/mnt/k8s-disks/0` (`*-mx-s3-local`)
  - UI: three-branch mermaid diagram, plus a comparison table (load speed, cost, setup effort).

UI: diagrams carry the section, with only brief prose. Deep dives go in `<details>`.

---

## 3. When is Dynamo a good fit? ✅ [NEW]

- 3.1 Good fit:
  - Large or MoE models spanning multiple GPUs or nodes
  - High concurrency or long-context workloads where prefill dominates
  - Strict TTFT/ITL SLAs
  - Customer wants to own the stack on EKS
- 3.2 Probably not the right fit:
  - Small model, single GPU, low QPS
  - No Kubernetes skills on the team
  - Customer wants fully managed inference → point to Bedrock or SageMaker
- 3.3 Comparison table: Dynamo on EKS vs plain vLLM on EKS vs managed options. Columns: operational effort, performance tuning, scale, cost control.
- 3.4 Discovery questions: model, traffic shape (ISL/OSL, concurrency), SLAs, GPU access, team skills.

UI: `Tabs` (Good fit / Not a fit / Compare). The discovery questions go in an info `Admonition`.

---

## 4. Prerequisites and platform install 🚀

- 4.1 Prerequisites:
  - GPU quota or Capacity Blocks
  - Tools: AWS CLI, kubectl, terraform, helm, jq, envsubst
  - Hugging Face token
- 4.2 Deploy infrastructure: clone the repo, review `blueprint.tfvars` (show the full file from Erez's commit), run `install.sh`.
- 4.3 Prepare the environment, in this order:
  1. Configure kubeconfig
  2. Wait for ArgoCD to report Healthy
  3. Export `S3_BUCKET`, `MODEL_SYNC_SA`, `INFERENCE_SA`
  4. Create the ServiceAccounts
  5. Create the HF secret
  6. Create the EFS PVCs (`model-cache`, `compilation-cache`, `perf-cache`)
  7. Verify
- 4.4 Model weight caching: S3 (recommended) vs EFS PVC.
  - **[NEW][ERZ]** Sync parameters and measured times per model:

    | Model | Safetensors files | `fileWorkers` | `storageSize` | Measured sync |
    |---|---|---|---|---|
    | Qwen3-32B | 18 × ~4 GB | 18 | 100 GiB | 54 s |
    | GLM-5.3-Flash | 62 × ~5.4 GB | 30 | 200 GiB | 184 s |
    | GLM-5.3 | 141 × ~5.5 GB | 30 | 200 GiB | 413 s |
    | Nemotron-3-Ultra | 225 × ~5 GB | 35 | 200 GiB | 943 s |

  - **[NEW][ERZ]** Sizing rule: `storageSize ≈ largest file × fileWorkers`.
  - **[ERZ]** Caution `Admonition` for a sync pod stuck Pending. Causes: instance type not available in region/AZ, vCPU quota, NodePool restrictions. Relax constraints in this order: `requireNetworkBandwidth` → `storageSize` → `fileWorkers`.
- 4.5 ModelExpress (optional): CRDs and RBAC, plus the P2P transfer sequence diagram.

UI:
- `ProgressSteps` bar (Prereqs → Deploy → Prepare → Cache weights), repeated at the top of each sub-step.
- All install steps go in one `<details>` block.
- 4.4 uses two `Tabs` (S3 / EFS).
- 4.5 uses `Tabs` (CRDs / RBAC), plus a collapsed "How P2P works" section.

---

## 5. Production-ready blueprints by model size and instance 🧾

- 5.1 Blueprint picker, one row per manifest **[ERZ]**:

  | Size | Model | Instance | Manifests | Mode |
  |---|---|---|---|---|
  | Small | Nemotron-3.5-Bolt-E4B | g6e | (upstream only, see open items) | agg |
  | Medium | Qwen3-32B | p5.48xlarge H100, TP4 | `h100-agg`, `h100-agg-mx-s3`, `h100-disagg` | agg + disagg |
  | Medium | GLM-5.3-Flash | g7e.48xlarge, TP8 | `g7e-agg-mx-s3`, `g7e-agg-mx-s3-local` | agg |
  | Large | GLM-5.3 | p6-b300.48xlarge, TP8 | `b300-agg-mx-s3`, `b300-agg-mx-s3-local` | agg |
  | Large | Nemotron-3-Ultra 550B | 1P + 1D × p6-b300, TP8 each | `b300-disagg-mx-s3` | disagg |

  - **[NEW][ERZ]** Image and version matrix:
    - Frontend: `dynamo-frontend:1.6.0`
    - Nemotron worker: `vllm-runtime:1.6.0-efa`
    - `modelexpress-server:0.7.0`
    - GLM recipes: custom `${IMAGE}`, plus `runtimeVersionOverride: 1.5.1`
- 5.2 How to read manifest names: instance, topology, `mx` (ModelExpress), source (`s3`, `s3-local`, or PVC by default).
- 5.3 **Aggregated serving**
  - When to use it: the default, and simpler to run.
  - Walkthrough: Qwen3-32B `h100-agg-mx-s3`.
  - **[ERZ]** Extra tab for GLM-5.3: watch source vs target worker logs using the `grove.io/podclique-pod-index` selectors.
  - Tabs: Deploy · Manifest · Smoke test · Scale out (ModelExpress P2P) · Remove.
- 5.4 **Disaggregated serving**
  - When to use it: long prompts, prefill-heavy traffic, scaling prefill and decode independently.
  - Requirements: EFA, workers in the same AZ (required zone podAffinity), NIXL with LIBFABRIC, sizing prefill vs decode.
  - **[ERZ]** Main walkthrough: Nemotron-3-Ultra `b300-disagg-mx-s3`:
    - 1M context, FP8 KV cache
    - `NixlConnector` (LIBFABRIC)
    - KV events from the prefill worker
    - 8 GPUs + 16 EFA per worker
  - Smaller alternative: Qwen3-32B `h100-disagg` (EFS PVC).
  - Tabs: Deploy · Manifest · Verify KV transfer · Smoke test · Remove.
- 5.5 **[NEW][ERZ]** Reasoning and tool-call parsers per model (`qwen3_coder`, `nemotron_v3`, `glm47`, `glm45`), plus a tool-calling smoke-test tab.
- 5.6 **[NEW][ERZ]** Production settings checklist:
  - Frontend and ModelExpress on on-demand capacity
  - Workers required to share a zone
  - `IPC_LOCK`
  - `sharedMemorySize` of 50–100 Gi
  - `startupProbe` `failureThreshold: 540` (~90 min) and `VLLM_ENGINE_READY_TIMEOUT_S=5400`, to set expectations for large-model startup
- 5.7 Links to the per-model recipe READMEs.

UI:
- 5.1 is grouped into `Tabs` (Small / Medium / Large).
- 5.3 and 5.4 use the same `Tabs` layout so they're easy to compare.
- A tip `Admonition` at the top of 5.4: "Start with aggregated. Move to disaggregated when profiling shows a prefill bottleneck."
- 5.6 goes in a `<details>` checklist.

---

## 6. Startup and scale-out performance ⚡ [NEW][ERZ]

Source: the `nvidia-dynamo-modelexpress-startup-timing` skill in Erez's commit.

- 6.1 The six startup phases: boot and registration → imports and config → worker fork + NCCL → weight load → compile, warmup, KV cache, autotune, CUDA graphs → ready.
- 6.2 Cold vs warm start, with headline numbers observed in Erez's runs:
  - ~12.3 min cold vs ~3.4 min with warm compile artifacts
  - DeepGEMM warmup ~275 s cold (GLM-5.3 FP8 MoE, TP8)
  - S3 stream ~67 s (GLM-5.3, TP8)
  - P2P RDMA ~1.18 Tbps effective node rate (GLM-5.3-Flash, TP8, g7e)
- 6.3 Caches to seed: `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, FlashInfer autotune cache, `MX_ARTIFACT_TRANSFER`, the `compilation-cache` PVC, and a persistent DeepGEMM cache.
- 6.4 Source vs receiver workers: how to tell them apart in logs, and what `Source publish attempt failed` means.
- 6.5 Measure it yourself: the `[TIMING]` grep commands, plus a pointer to the agent skill.

UI:
- Mermaid timeline of the phases.
- `StatusGrid` for the headline numbers.
- Grep commands in `CodeBlock`s inside `<details>`.

---

## 7. Profiler: is your deployment optimal? 📊 [NEW]

- 7.1 What the profiler does: sweeps TP and prefill/decode splits against target TTFT/ITL and recommends a configuration.
- 7.2 Define SLA targets: ISL/OSL, TTFT, ITL, concurrency.
- 7.3 Run the profiler on EKS: job or manifest, where results land (`perf-cache` PVC), expected duration.
- 7.4 Read the results: throughput vs latency, recommended agg/disagg layout.
- 7.5 Benchmark the live endpoint: load test (for example AIPerf), plus key Prometheus/Grafana metrics.
- 7.6 Apply the recommendation: map it back to a blueprint manifest.

UI:
- `ProgressSteps` (Define SLA → Profile → Review → Apply)
- An example results table or image
- Commands in `CodeBlock`s

---

## 8. Planner (phase 2) 🔮

- 8.1 What the Planner does: SLA- and load-based autoscaling of prefill and decode workers.
- 8.2 How it relates to the Profiler: it uses the profiling data.
- 8.3 Status: "Coming soon", with a link to the Dynamo Planner docs.

UI: a single note `Admonition` as a placeholder, so the sidebar entry exists now.

---

## 9. Observability and troubleshooting 🩺

- Prometheus/Grafana dashboards (kube-prometheus-stack is enabled in the tfvars).
- Common symptoms:
  - ArgoCD application not Healthy
  - **[ERZ]** Sync pod stuck Pending (Karpenter, quota, NodePool)
  - Capacity Block not used (missing `ai-on-eks=true` tag)
  - EFA/NIXL errors, or workers placed in different AZs
  - **[ERZ]** `503` from `tower_http` during startup: expected before the worker is ready
  - **[ERZ]** `No available shared memory broadcast block found in 60 seconds`: expected during long warmups
  - **[ERZ]** Worker serving but not exposing itself as a P2P source (source publish failed)

UI: accordion of `<details>` blocks, one per symptom.

---

## 10. Cleanup 🧹

- `Tabs`: Remove model deployments / Destroy infrastructure.
- Keep the warning that the S3 bucket and its weights are deleted.

---

## Next steps and resources

- NVIDIA Dynamo docs, Dynamo GitHub, ModelExpress GitHub
- Recipe READMEs, `model-cache/README.md`
- Related AI on EKS inference blueprints

---

## Open items

- [ ] **Profiler (7) and Planner (8):** not covered in Erez's commit. Source the content from the Dynamo docs or new testing. The commands, CRDs, and tools in 7.3–7.5 are placeholders.
- [ ] **Nemotron-3.5-Bolt-E4B on g6e:** listed upstream but not in Erez's commit. Confirm its source, since it's the only "Small" tier recipe.
- [ ] **GLM `${IMAGE}`:** placeholder in the GLM recipes. Get the actual image from Erez.
- [ ] **Model size buckets:** confirm GLM-5.3 and GLM-5.3-Flash parameter counts.
- [ ] **Caching and ModelExpress placement:** keep them under Prerequisites (4.4–4.5), or move them to a "Common setup" block at the top of section 5.
- [ ] **Section 9:** my addition, not in the original outline. Keep or drop.
- [ ] **Korean translation:** `website/i18n/ko/docusaurus-plugin-content-docs/current/blueprints/inference/framework-guides/GPUs/nvidia-dynamo.md` exists in the commit. Update it, or add a follow-up note.
