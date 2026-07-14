# Cosmos 3 Serving — Latency Attribution & Optimization

**Project:** `cosmos-serving`
**Owner:** Anastasiia Alekseeva
**Status:** Draft spec — **Part 1 (reproduce NVIDIA + selectable `optimize`)**; **Part 2 (1–2 new techniques)** later
**Last updated:** 2026-07-13

---

## 0. Refinement (2026-07-13) — realign to what §5.3 actually quantifies

The first pass inverted the report's emphasis and over-built the harness. Corrected here:

- **The report quantifies the *Generator*, not the Reasoner.** Every stated serving
  number in §5.3 is Generator-side: CUDA graphs **30–60% on T2I** (§5.3.1), CFG-Parallel
  "nearly halves per-step latency", reasoner-tower caching, Cache-DiT, FP8, VAE-patch,
  and request batching **Table 9**. The Reasoner (§5.3.2) is explicitly "reuse Qwen3-VL
  in vLLM/TRT-LLM out of the box" — the report presents **no reasoner technique ablation**.
- **So the Reasoner gets a stock-vLLM concurrency/shape sweep, not a technique waterfall.**
  The earlier reasoner ladder (crippled `--enforce-eager --no-enable-prefix-caching
  --max-num-seqs 1` baseline, each rung re-enabling a vLLM default) measured stock-vLLM
  defaults, not NVIDIA techniques. Replaced by TTFT + throughput vs concurrency
  {1,64,128,256} at fixed shapes (mirrors `inference_benchmarks.md`).
- **Batching is a throughput result, not a latency rung.** It is measured in a separate
  batching-throughput sweep that reproduces **Table 9**, not placed on the latency waterfall.
- **Harness cut to the faithful core.** Removed: the eager parallel reasoner ladder,
  the FA2×compile probe (a self-inflicted rabbit hole — the report doesn't toggle
  flash-attn), roofline, per-stage timers, the equivalence stub, and the workflow twin.
  Kept: workload OPs, the ablation runner, the sweep runners, both backends, plots.
- **Superseded below:** the Reasoner ladder (old §6) and the reasoner rows of §4/§5/§11.
  §6 is rewritten. Sections not touched still describe the general method.

---

## 1. Goal

**Part 1 — Reproduce & attribute.** Reproduce NVIDIA's Cosmos 3 serving optimizations on **Nebius**, for **both towers**, packaged as an **in-repo `optimize` command where you select individual techniques or the full bundle**. The Nebius **workbench** (`npa`) is used only to **provision the infrastructure** that runs it (a serverless GPU job + the optimized-model endpoint) — it does not implement the optimization logic. Show **how each technique contributes to latency** via a contribution waterfall.

**Part 2 — Improve.** Introduce **just one or two new techniques** on top to push latency further, and prove their contribution on the same plot.

**Order of work:** **Reasoner first** (fast to iterate, clean attribution), then **Generator** (biggest absolute latency), then Part 2.

The key design unifier: the **selectable-technique interface and the ablation waterfall are one mechanism** — the waterfall is the `optimize` command run over cumulative technique subsets (§5). Each technique is a toggle; `--preset full` = all of NVIDIA's; `--preset none` = naïve baseline.

---

## 2. Background (decisions this spec rests on)

From the Cosmos 3 technical report (arXiv 2606.02800) and prior analysis:

- **Two towers, both in scope:**
  - **Reasoner** (VLM, autoregressive, **Qwen3-VL** backbone) — fast: ~83 ms TTFT, ~2,800 tok/s at high concurrency. Served by **vLLM**. Runs on **1× H200**.
  - **Generator** (diffusion, MoT) — slow: ~108 s (Nano 720p, 1× B200); ~240 s on 1× H200. Served by **vLLM-Omni**. Runs on **2× H200** (to enable CFG-Parallel).
- **What Part 1 reproduces & attributes** (see §0 for why the Reasoner differs):
  - *Reasoner:* **no technique toggles** — served by stock vLLM (paged KV-cache, continuous batching, fused attention, prefix caching, all default-on; §5.3.2). Characterized by a concurrency/shape sweep (TTFT + throughput), not an ablation. *(The report lists neither EVS nor "deferred sampling sync" in §5.3; both were dropped.)*
  - *Generator:* reasoner-tower output caching, torch.compile + CUDA graphs, **CFG-Parallel** (cond/uncond on 2 GPUs), **Ulysses Context-Parallel** (alt 2-GPU strategy), Cache-DiT, FP8 quant, VAE-Patch-Parallel, request batching, **HSDP** + **CPU offload** (memory). All selectable toggles; the latency waterfall shows only the latency-reducing subset, batching is a throughput sweep (see §6).
- **Part-2 new techniques:** 1–2 only, **lossless preferred** (any lossy one gets a quality guard). Candidates picked from the Part-1 breakdown (§14).

### Open-source components to reproduce
| Component | Repo | License |
|---|---|---|
| Reasoner serving | upstream `vllm` (Qwen3-VL path) | Apache-2.0 |
| Generator serving | `vllm-project/vllm-omni` (recipe `recipes/cosmos3/Cosmos3-Nano.md`) | Apache-2.0 |
| PyTorch reference inference (readable ablation path) | `NVIDIA/cosmos-framework` | OpenMDW-1.1 (verify) |
| Weights | HF `nvidia/Cosmos3-Nano*` | OpenMDW-1.1 |
| Deployment / infra provisioning | `nebius/nebius-physical-ai` (`npa workbench cosmos`) | Apache-2.0 |

---

## 3. Scope & phasing

- **Model:** Cosmos 3 **Nano (16B)** — fastest iteration; NVIDIA's designated robotics-inference variant.
- **Hardware (split by tower):** Reasoner **1× H200 141 GB**; Generator **2× H200** (single-GPU techniques use 1 of the 2; CFG-Parallel uses both).

### Part 1 — Reproduce (Reasoner → Generator)
- **Reasoner (now):** robot task = next-action / visual reasoning (short video clip + task prompt → short reasoning/action output). Reproduce + attribute the Reasoner techniques (§6).
- **Generator (next):** T2V + I2V (I2V ≈ action-conditioned world-model rollout). Reproduce + attribute the Generator techniques incl. CFG-Parallel (§6). 256p/480p for iteration, 720p for the headline number.
- **Deliverable:** the selectable, in-repo `optimize` command (§4/§5/§8) + contribution waterfall + stage breakdown per tower.

### Part 2 — Improve (later)
- Add **1–2 new techniques** targeting the dominant stage the Part-1 breakdown reveals; prove lossless (or guard if lossy); append to the same waterfall.

### Non-goals
- Training / fine-tuning.
- Stacking two distributed strategies at once (CFG-Parallel × Context-Parallel needs ≥4 GPUs; on 2× H200 they are mutually-exclusive alternatives).
- Chasing quality-benchmark leaderboards (Physics-IQ / RoboLab / VANTAGE-Bench) — small correctness guards only, never the working loop.

---

## 4. Deliverables

1. **Selectable, in-repo `optimize`** (Part 1) — per-technique toggles + `--preset full|none`, for both towers, implemented in this repo (`optimize/cli.py`); the workbench only provisions the Nebius infra that runs it (§5, §7, §8).
2. **Technique-contribution waterfall** (per tower) — cumulative ablation, each bar = one technique's marginal latency contribution; `hw2/ablation.py` "vs V0 / vs prev" rendered as a waterfall. Part 2 appends new-technique bars.
3. **Stage breakdown** (per tower) — where wall-clock goes, so each technique's win traces to the stage it shrinks.

**Reasoner stages:** input prep/tokenize · **vision encode (ViT)** · **EVS pruning** (savings) · prefill (→ TTFT) · decode/token (→ TPOT) · sampling/detokenize.
**Generator stages:** input prep · (optional) prompt upsampling · reasoner conditioning (cached) · **denoising loop** `N_steps × 2 forwards (CFG)` → {attention, FFN, AdaLN, CFG cost / **CFG-Parallel P2P sync**, host launch overhead, Cache-DiT savings} · sampler update · **VAE decode** · guardrails · output encode.

**Acceptance:** summed stages reconcile to the black-box `time_generation()` wall-clock within ≤5%, and agree with the torch.profiler / Nsight trace on dominant terms.

---

## 5. The selectable optimization interface (Part-1 core)

One mechanism serves reproduction and attribution. The command lives in this repo
(`python -m optimize.cli`); the workbench only provisions the Nebius infra to run it (§7).

### Technique toggles + presets
Each NVIDIA technique (§6) is an independent switch; presets bundle them.

```
# Generator: full package / naïve baseline / hand-picked subset
python -m optimize.cli --tower generator --preset full
python -m optimize.cli --tower generator --preset none
python -m optimize.cli --tower generator \
    --enable reasoner-cache,cuda-graphs,cache-dit,fp8,vae-patch,cfg-parallel
# Generator: cumulative latency waterfall + Table 9 batching throughput
python -m optimize.cli --tower generator --ablate --out-dir results
# Reasoner: stock-vLLM concurrency/shape sweep (no toggles; §0)
python -m optimize.cli --tower reasoner --out-dir results
```

- **Generator only:** `--preset full` = every Generator technique; `--preset none` = naïve baseline; `--enable a,b,c` = explicit subset.
- `--ablate` (Generator) walks the cumulative latency ladder (§6a) → contribution waterfall, and runs the batching sweep (§6b).
- **Reasoner** takes no toggles — it runs the concurrency sweep (§6c).
- Lossy techniques (FP8, Cache-DiT) auto-trigger the quality guard (§ below).

### Benchmark & workload — *fast to run, faithful to real latency*
**Serving latency is shape-driven, not content-driven.** TTFT/TPOT/E2E depend on `(input_len, output_len, concurrency, #multimodal_tokens)` — not the actual words/pixels. So accuracy comes from **fixed representative shapes**, not a big dataset (that's only an occasional guard). This mirrors NVIDIA's `inference_benchmarks.md` (fixed `(in,out)` tokens × concurrency `1/64/128/256` → TTFT / latency / throughput).

**Attribution needs the regime where each technique acts** — so we sweep input/output
shape, not a single point (see §6 for the exact OPs):

| Experiment | Shapes swept | Reveals |
|---|---|---|
| **Generator latency waterfall** | T2I-1024; T2V 256p/480p/720p; I2V-480 | CUDA graphs (T2I), reasoner-cache, Cache-DiT, FP8, VAE-patch, CFG-Parallel |
| **Generator batching throughput** | T2V 256p (B≤6), 480p (B≤3) | request batching (Table 9) |
| **Reasoner concurrency sweep** | text & video shapes × concurrency 1/64/128/256 | stock-vLLM TTFT + throughput (paged KV-cache, continuous batching) |

Report **median + p95** over ≥N repeats after warmup (exclude init/compile).

**Drivers (ready-made):** vLLM-Omni timed generation (Generator); **NVIDIA AIPerf** (Reasoner sweep — TTFT + throughput at each concurrency).

**Quality guard (occasional, not the loop):** lossless techniques → exact-match / tight numerical-equivalence; lossy (FP8, Cache-DiT) → a tiny fixed slice checked once per variant for acceptable drift.

**Net:** the Generator waterfall (6 techniques × 5 OPs × N) plus the batching and reasoner sweeps run in **minutes-to-hours on one/two GPUs** — fast, because shapes are fixed.

---

## 6. Experiments (rewritten per §0)

Three figures, mapped 1:1 onto the report's stated numbers. The `optimize` interface
still allows any Generator technique subset via `--enable` / `--preset full`.

### 6a. Generator latency waterfall (Part 1b) — the report's headline story
Cumulative order (baseline = naïve PyTorch: recompute conditioning + sequential CFG).
OPs: **t2i-1024 / t2v-256 / t2v-480 / i2v-480 / t2v-720**. Different rungs dominate at
different points, which is the point of sweeping input/output shape.

| # | + technique | Category | Anchor / note |
|---|---|---|---|
| G0 | naïve PyTorch reference | — | recompute conditioning, sequential CFG |
| G1 | + **reasoner-tower output caching** | latency | conditioning invariant across steps → compute once |
| G2 | + **torch.compile / CUDA graphs** | latency | **30–60% on T2I** (§5.3.1); host-launch-bound, fades on video |
| G3 | + **Cache-DiT** *(lossy→guard)* | latency | reuse cached block outputs; more steps → more win |
| G4 | + **FP8 quant** *(lossy→guard)* | latency | memory-bound denoise |
| G5 | + **VAE-Patch-Parallel** | latency | shrinks decode tail (bigger at high-res) |
| G6 | + **CFG-Parallel (2 GPU)** | scaling | "nearly halves per-step latency" (§5.3.1). Bar = a 2nd GPU → labeled scaling (N4). **End of Part 1.** |
| **P2** | **+ 1–2 NEW techniques (Part 2)** | — | picked from the breakdown; lossless preferred (§14) |

Selectable but **off the latency waterfall** (`--enable` / `--preset full`, not `--ablate`):
**Ulysses Context-Parallel** (alt 2-GPU strategy, mutually exclusive with CFG-Parallel),
**request batching** (throughput — see 6b), **HSDP** + **CPU offload** (memory; CPU
offload *adds* latency).

### 6b. Generator batching throughput (Table 9)
Batching amortizes per-step overhead → throughput, not per-clip latency, so it is a
separate sweep: throughput at B=1 vs B=`batch_max` on **T2V 189-frame** at 256p
(B≤6) and 480p (B≤3). Reproduces **Table 9** (256p 8–55%, 480p 1–5%). 720p is omitted —
the 74k-token context admits only B=1.

### 6c. Reasoner concurrency/shape sweep (Part 1a) — 1× H200
No technique ladder (§0). Stock vLLM (paged KV-cache, continuous batching, fused
attention, prefix caching all on by default), swept over **concurrency {1,64,128,256}**.
**1:1 with `inference_benchmarks.md`:** fixed **input=50** tokens, output **1**
(captioning → request-latency / req-s regime) and **100** (VQA → token-throughput
regime), BF16 / batch-1, measured with **AIPerf**. Report **TTFT · request latency ·
tok/s · req/s** per point, faceted by output length. NVIDIA benchmarks only *video*
(1 & 2 FPS) — those are reproduced exactly; **text** and **image** inputs are added
for coverage (4 modality families × 2 outputs × 4 concurrencies = 32 points).
*VERIFY on-box:* NVIDIA does not publish the clip duration/resolution or the AIPerf
media flags, so the FPS→frame counts and video/image input config are best-effort
(see `# VERIFY` in `bench/aiperf.py` / `bench/workload.py`).

---

## 7. Framework & infrastructure

| Concern | Choice | Rationale |
|---|---|---|
| **Reasoner serving** | **vLLM** (Qwen3-VL path), 1× H200 | Paged KV-cache, continuous batching, fused attention out of the box. |
| **Generator serving** | **vLLM-Omni**, 2× H200 | NVIDIA's Generator serving approach; 2 GPUs enable CFG-Parallel. |
| **Readable ablation / build techniques** | **cosmos-framework PyTorch reference path** | Report §5.3.1: primary target for new features, validated first here. Eager, per-stage timers. |
| **Latency benchmark driver** | vLLM bench / **GenAI-Perf**; `hw2 time_generation` for eager | Standard, fast, reproducible (§5). |
| **Deployment / infra provisioning** | Nebius `npa workbench cosmos` + MLflow | Ops shell; provisions the serverless GPU job that runs our repo `optimize` command + the optimized-model endpoint + the quality-guard job. |
| **Instrumentation** | Adapted from `gpu_and_inference_hw` (§9) | Timing, profiling, ablation, roofline, plots. |

### Workbench = infrastructure provisioning only (do NOT implement `npa optimize`)
The `optimize` command lives in **this repo** (`optimize/cli.py`, run via `python -m optimize.cli`),
not in the workbench. npa's built-in `optimize_cmd` slot (a `typer.echo("not yet implemented")`
placeholder for "TensorRT compilation and quantization") is left **untouched** — we deliberately
do not depend on or fill it. The workbench is used purely to provision Nebius infrastructure:
- **Serverless GPU job:** an `npa.workflow` twin (+ `skypilotTwin`) that provisions the GPUs,
  clones this repo, and runs its `optimize.cli` — see `jobs/cosmos3-ablation.*`. There is **no
  `toolRef` to a workbench optimize tool**; the job just runs repo code and publishes artifacts.
- **Optimized-model endpoint:** `npa workbench cosmos deploy --runtime serverless` with the
  ablation-winning engine args passed through (`--extra-serve-args`) — see
  `jobs/deploy-optimized.sh` / `workbench/optimized-deploy.config.yaml`.
- **Quality-guard job:** the RoboLab-120 eval as a second provisioned job — `jobs/robolab-eval.*`.
- **Artifacts:** the repo's `optimize.cli` emits waterfall / breakdown JSON to the results
  bucket (`s3://serverless-challenge/cosmos3-ablation-results/`), with schema versioning.

---

## 8. Requirements

### Functional
- **F1** Deploy Nano: Reasoner via vLLM (1× H200), Generator via vLLM-Omni (2× H200), + cosmos-framework eager path for readable ablation.
- **F2** Fixed, reproducible latency benchmark (§5): synthetic fixed-shape workload, 4 OPs, warmup excluding init/compile, median + p95, MLflow.
- **F3** **Selectable optimization** (§5): per-technique toggles + `--preset full|none` + `--ablate`, both towers.
- **F4** Cumulative ablation → "vs V0 / vs prev" table → contribution waterfall.
- **F5** Per-stage instrumentation (§4) + kernel trace export (torch.profiler → Perfetto; Nsight); breakdown reconciles ≤5%.
- **F6** Roofline classification of dominant stages (compute- vs memory-bound).
- **F7** In-repo `optimize` command (§5); the workbench provisions the infra to run it — a serverless GPU job + the optimized-model endpoint + the quality-guard job (§7).
- **F8 (Part 2)** New technique(s): numerical-equivalence test where lossless (SSIM≈1.0 / exact-match); appended to the same waterfall with marginal contribution.

### Non-functional
- **N1 Reproducibility:** every number reproducible from one command + logged config (MLflow).
- **N2 Comparability:** methodology mirrors `inference_benchmarks.md` (fixed shapes; TTFT/latency/throughput; Generator 189 frames @24 FPS, BF16).
- **N3 Regime coverage:** each technique measured in its operating point (§5).
- **N4 Attribution honesty:** by-difference & architectural techniques labeled; **CFG-Parallel's bar = adding a 2nd GPU** (scaling, not per-GPU algorithmic win) — labeled as such.
- **N5 Hardware:** Reasoner 1 GPU; Generator ≤2 GPU (2 only when CFG-Parallel enabled).

---

## 9. Reused assets from `gpu_and_inference_hw`

| Asset | Source | Use here |
|---|---|---|
| **`ablation.py` cumulative-variant pattern** + "vs V0 / vs prev" table + trace-per-variant + `_Tee` | `hw2/ablation.py` | **Core method** for both ladders (§6) → the waterfall + the `--ablate` mode. |
| `time_generation(loop_fn, …)` | `hw2/utils.py` | Ground-truth latency for eager variants (R0–R5, G0–G3). |
| `profile_variant()` — torch.profiler → Chrome trace | `hw2/ablation.py` | Kernel trace per variant; validates stage attribution. |
| `compute_stats()` (TTFT/p95, E2E, throughput), `print_stats()` | `hw3/engine_utils.py` | Reasoner serving metrics (Part 1a). |
| `plot_results()` / `plot_policy_results()` | `hw3/engine_utils.py` | Template for waterfall + before/after figures. |
| `generate_workload()` (synthetic, shared prefix) | `hw3/engine_utils.py` | Basis for OP-C concurrency workload. |
| CUDA-event timing, `GPU_SPECS`, `measure_roofline_points`, `plot_roofline`, `save_roofline_data` | `hw1/` | Roofline (F6). **Extend `GPU_SPECS` with BF16/FP8 tensor-core peaks + HBM BW for H200 (4.8 TB/s).** |
| VM scripts (`01_…`–`05_…`, `config.sh`) | `scripts/` | Nebius VM create / upload / run / fetch. |

**To build (thin):** OP-matrix workload gen + GenAI-Perf/vLLM-bench wrappers; the technique-toggle registry + `--ablate`; per-stage CUDA-event timers; `plot_contribution_waterfall()`, `plot_stage_breakdown()`.

---

## 10. Observability (prefer ready solutions; build only the thin layer)
- **Kernel (offline):** torch.profiler → Perfetto; **Nsight Systems** for GPU timeline (+ CFG-Parallel P2P). → drives stage breakdown.
- **Serving (online):** vLLM / vLLM-Omni **Prometheus `/metrics`** (TTFT, TPOT, queue, throughput) → Prometheus + Grafana.
- **GPU (online):** **DCGM-Exporter** → Grafana (SM occupancy, HBM BW util, NVLink for CFG-Parallel, power).
- **Tracking:** **MLflow** (in the workbench) — config, metrics, PNGs per run.
- **Build ourselves:** per-stage timers + the two plot functions only.

---

## 11. Implementation steps

### Part 1a — Reasoner (now), 1× H200
1. Provision Nebius 1× H200 VM (adapt `scripts/`); install vLLM + cosmos-framework eager path; auth; pull Nano weights.
2. Smoke-test Reasoner (short clip + prompt → reasoning) on eager + vLLM; fix seed.
3. Build the §5 benchmark: OP matrix (A–D), fixed shapes, warmup, median/p95, MLflow; wrap GenAI-Perf/vLLM-bench; `time_generation` for eager.
4. Build the **technique-toggle registry + `--ablate`**; run the Reasoner ladder (R0→R8) → "vs V0 / vs prev".
5. Stage-instrument the dominant variant + traces; reconcile ≤5%; roofline-classify.
6. Ship the Reasoner **contribution waterfall + stage breakdown + findings**; quality-guard FP8/EVS.
7. Wire the Reasoner path into the repo's `optimize` command (§5); make it provisionable as a Nebius job (§7).

### Part 1b — Generator (next), 2× H200
8. Deploy Generator via vLLM-Omni + eager path; reproduce NVIDIA 256p/480p/720p numbers.
9. Run the Generator ladder G0→G7 (incl. **CFG-Parallel on 2 GPUs**); build waterfall + breakdown; quality-guard Cache-DiT/FP8; extend `optimize` to `--tower generator`.

### Part 2 — Improve (later)
10. Pick **1–2 new techniques** for the dominant stage (candidates §14); implement in the eager path first, then vLLM-Omni.
11. Prove lossless (F8); append to the same waterfall; report marginal contribution + E2E delta.
12. Ship via the repo's `optimize` command + the Nebius job/deploy specs (`jobs/`); capture Grafana/MLflow evidence.

---

## 12. Success criteria
- **Part 1:** an in-repo `optimize` command selecting any technique subset or `--preset full` for both towers; contribution waterfalls attributing each NVIDIA technique's saving across the four regimes, reconciling ≤5% to wall-clock; end-to-end within a documented margin of NVIDIA's numbers.
- **Part 2:** 1–2 new techniques with a measurable marginal contribution on the same waterfall; lossless proven (or drift reported for a guarded lossy one); no regression on the quality guard.

---

## 13. Decisions (resolved)
- Model **Cosmos 3 Nano**; **Reasoner 1× H200**, **Generator 2× H200**.
- OP-D input = **short video clip**; **vLLM** (TensorRT-LLM deferred).
- **Part 1 = reproduce all NVIDIA techniques** behind a **selectable, in-repo `optimize` command** (per-technique or full package); CFG-Parallel included on 2 GPUs.
- **Part 2 = 1–2 new techniques.** *(Which ones: picked from the Part-1 breakdown — see §14.)*
- `npa optimize` stays an upstream placeholder — **not** implemented here; the workbench only provisions infra (§7).

## 14. Part-2 candidate techniques (decide after the Part-1 breakdown)
Lossless-preferred levers not covered by NVIDIA's set:
- **CFG cond/uncond batched B=2** on a single rank (weight-read amortization — complements or replaces CFG-Parallel).
- **Async/overlapped CFG-Parallel** — hide the once-per-step P2P behind compute (headroom is small; NVIDIA already runs the two passes concurrently).
- **VAE decode** optimization beyond patch-parallel.
- **Cache-DiT composed with fuller CUDA-graph capture.**
- **Attention/FFN kernel fusion** on the DiT block.

## 15. Risks
- **CFG-Parallel = reproduction, not novelty:** G7 is NVIDIA's technique; the genuine new contribution is Part 2. Keep that framing explicit.
- **Cross-hardware attribution:** G7's bar mixes "one more GPU" with "CFG split" — label as a scaling technique, don't present as a per-GPU algorithmic win (N4).
- **Cross-backend attribution:** eager (R0–R5) vs vLLM (R6+) aren't one codebase; architectural techniques attributed engine-vs-eager.
- **Regime mis-attribution:** enforce N3 (each technique in its OP).
- **Lossy-guard drift:** FP8/EVS/Cache-DiT may shift outputs; report drift honestly.
- **License:** confirm OpenMDW-1.1 terms for `cosmos-framework` before productizing.

## 16. Proposed repo layout
```
cosmos-serving/
  specification.md
  deploy/            # Nebius VM + npa workbench glue (from scripts/)
  serving/           # vLLM (reasoner, 1 GPU) + vLLM-Omni (generator, 2 GPU) + eager-path configs
  bench/
    workload.py      # OP matrix (A–D) synthetic fixed-shape generator (§5)
    drivers.py       # GenAI-Perf / vLLM-bench wrappers + time_generation
    ablation.py      # cumulative technique ladders + --ablate (§6)  [from hw2/ablation.py]
    stages.py        # per-stage CUDA-event timers (§4)
    roofline.py      # extended GPU_SPECS + classification  [from hw1]
    plots.py         # plot_contribution_waterfall() + plot_stage_breakdown()  [from hw3]
    equivalence.py   # Part-2 lossless test (token/latent diff, SSIM)
  optimize/          # engine techniques + toggle registry — the repo's optimize command
    techniques/      # one module per toggle (kv_cache, cuda_graphs, cfg_parallel, cache_dit, fp8, vae_patch, evs, …)
    registry.py      # technique registry + presets (none/full) + --ablate order
    cli.py           # optimize CLI (python -m optimize.cli)
  workbench/         # Nebius INFRA provisioning configs (deploy the optimized endpoint; serve DROID) + runbook
  jobs/              # Nebius job specs: serverless ablation job + optimized deploy + RoboLab-eval (+ npa.workflow twins)
  observability/     # Prometheus/Grafana/DCGM + dashboards
  results/           # traces, PNGs, JSON, MLflow artifacts
```
