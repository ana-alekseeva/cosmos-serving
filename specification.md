# Cosmos 3 Serving — Latency Attribution & Optimization

**Project:** `cosmos-serving`
**Owner:** Anastasiia Alekseeva
**Status:** Draft spec — **Part 1 (reproduce NVIDIA + selectable `optimize`)** scoped now, Reasoner→Generator; **Part 2 (1–2 new techniques)** later
**Last updated:** 2026-07-11

---

## 1. Goal

**Part 1 — Reproduce & attribute.** Reproduce NVIDIA's Cosmos 3 serving optimizations on **Nebius**, for **both towers**, packaged as a **workbench-compatible `optimize` command where you select individual techniques or the full bundle**. Show **how each technique contributes to latency** via a contribution waterfall.

**Part 2 — Improve.** Introduce **just one or two new techniques** on top to push latency further, and prove their contribution on the same plot.

**Order of work:** **Reasoner first** (fast to iterate, clean attribution), then **Generator** (biggest absolute latency), then Part 2.

The key design unifier: the **selectable-technique interface and the ablation waterfall are one mechanism** — the waterfall is the `optimize` command run over cumulative technique subsets (§5). Each technique is a toggle; `--preset full` = all of NVIDIA's; `--preset none` = naïve baseline.

---

## 2. Background (decisions this spec rests on)

From the Cosmos 3 technical report (arXiv 2606.02800) and prior analysis:

- **Two towers, both in scope:**
  - **Reasoner** (VLM, autoregressive, **Qwen3-VL** backbone) — fast: ~83 ms TTFT, ~2,800 tok/s at high concurrency. Served by **vLLM**. Runs on **1× H200**.
  - **Generator** (diffusion, MoT) — slow: ~108 s (Nano 720p, 1× B200); ~240 s on 1× H200. Served by **vLLM-Omni**. Runs on **2× H200** (to enable CFG-Parallel).
- **NVIDIA's shipped techniques = what Part 1 reproduces & attributes** (each a toggle in `optimize`):
  - *Reasoner:* KV cache, deferred sampling sync, torch.compile + CUDA graphs, fused/Flash attention, paged KV-cache, continuous batching, FP8/NVFP4 quantization, **EVS** (Efficient Video Sampling — token pruning for video inputs).
  - *Generator:* reasoner-tower output caching, torch.compile + CUDA graphs, **CFG-Parallel** (cond/uncond on 2 GPUs), Cache-DiT, FP8 quant, VAE-Patch-Parallel. (Ulysses CP / HSDP / CPU offload remain out of scope.)
- **Part-2 new techniques:** 1–2 only, **lossless preferred** (any lossy one gets a quality guard). Candidates picked from the Part-1 breakdown (§14).

### Open-source components to reproduce
| Component | Repo | License |
|---|---|---|
| Reasoner serving | upstream `vllm` (Qwen3-VL path) | Apache-2.0 |
| Generator serving | `vllm-project/vllm-omni` (recipe `recipes/cosmos3/Cosmos3-Nano.md`) | Apache-2.0 |
| PyTorch reference inference (readable ablation path) | `NVIDIA/cosmos-framework` | OpenMDW-1.1 (verify) |
| Weights | HF `nvidia/Cosmos3-Nano*` | OpenMDW-1.1 |
| Deployment / orchestration + `optimize` slot | `nebius/nebius-physical-ai` (`npa workbench cosmos`) | Apache-2.0 |

---

## 3. Scope & phasing

- **Model:** Cosmos 3 **Nano (16B)** — fastest iteration; NVIDIA's designated robotics-inference variant.
- **Hardware (split by tower):** Reasoner **1× H200 141 GB**; Generator **2× H200** (single-GPU techniques use 1 of the 2; CFG-Parallel uses both).

### Part 1 — Reproduce (Reasoner → Generator)
- **Reasoner (now):** robot task = next-action / visual reasoning (short video clip + task prompt → short reasoning/action output). Reproduce + attribute the Reasoner techniques (§6).
- **Generator (next):** T2V + I2V (I2V ≈ action-conditioned world-model rollout). Reproduce + attribute the Generator techniques incl. CFG-Parallel (§6). 256p/480p for iteration, 720p for the headline number.
- **Deliverable:** the selectable, workbench-compatible `optimize` command (§4/§5/§8) + contribution waterfall + stage breakdown per tower.

### Part 2 — Improve (later)
- Add **1–2 new techniques** targeting the dominant stage the Part-1 breakdown reveals; prove lossless (or guard if lossy); append to the same waterfall.

### Non-goals
- Training / fine-tuning.
- Distributed inference beyond CFG-Parallel (Ulysses CP, HSDP, CPU offload).
- Chasing quality-benchmark leaderboards (Physics-IQ / RoboLab / VANTAGE-Bench) — small correctness guards only, never the working loop.

---

## 4. Deliverables

1. **Selectable, workbench-compatible `optimize`** (Part 1) — per-technique toggles + `--preset full|none`, for both towers, dropping into Nebius's `npa … cosmos optimize` slot (§5, §8).
2. **Technique-contribution waterfall** (per tower) — cumulative ablation, each bar = one technique's marginal latency contribution; `hw2/ablation.py` "vs V0 / vs prev" rendered as a waterfall. Part 2 appends new-technique bars.
3. **Stage breakdown** (per tower) — where wall-clock goes, so each technique's win traces to the stage it shrinks.

**Reasoner stages:** input prep/tokenize · **vision encode (ViT)** · **EVS pruning** (savings) · prefill (→ TTFT) · decode/token (→ TPOT) · sampling/detokenize.
**Generator stages:** input prep · (optional) prompt upsampling · reasoner conditioning (cached) · **denoising loop** `N_steps × 2 forwards (CFG)` → {attention, FFN, AdaLN, CFG cost / **CFG-Parallel P2P sync**, host launch overhead, Cache-DiT savings} · sampler update · **VAE decode** · guardrails · output encode.

**Acceptance:** summed stages reconcile to the black-box `time_generation()` wall-clock within ≤5%, and agree with the torch.profiler / Nsight trace on dominant terms.

---

## 5. The selectable optimization interface (Part-1 core)

One mechanism serves reproduction, attribution, and the workbench.

### Technique toggles + presets
Each NVIDIA technique (§6) is an independent switch; presets bundle them.

```
# Full NVIDIA package
npa workbench cosmos optimize --tower generator --preset full
# Naïve baseline (all off)
npa workbench cosmos optimize --tower generator --preset none
# Hand-pick a subset
npa workbench cosmos optimize --tower generator \
    --enable reasoner-cache,cuda-graphs,cache-dit,fp8,vae-patch,cfg-parallel
npa workbench cosmos optimize --tower reasoner \
    --enable kv-cache,cuda-graphs,flash-attn,paged-kv,continuous-batching,fp8,evs
# Run the cumulative ablation and emit the waterfall + breakdown
npa workbench cosmos optimize --tower reasoner --ablate --output json
```

- `--preset full` = every NVIDIA technique for that tower; `--preset none` = naïve baseline; `--enable a,b,c` = explicit subset.
- `--ablate` walks the canonical cumulative ladder (§6) and produces the contribution waterfall — i.e. the ablation *is* this command over subsets.
- Lossy techniques (FP8/NVFP4, EVS, Cache-DiT) auto-trigger the quality guard (§ below).

### Benchmark & workload — *fast to run, faithful to real latency*
**Serving latency is shape-driven, not content-driven.** TTFT/TPOT/E2E depend on `(input_len, output_len, concurrency, #multimodal_tokens)` — not the actual words/pixels. So accuracy comes from **fixed representative shapes**, not a big dataset (that's only an occasional guard). This mirrors NVIDIA's `inference_benchmarks.md` (fixed `(in,out)` tokens × concurrency `1/64/128/256` → TTFT / latency / throughput).

**Accurate attribution needs the regime where each technique acts** — one operating-point matrix, not one point:

| OP | Shape | Reveals |
|---|---|---|
| **A — latency** | 1 req, short in/out | CUDA graphs, deferred sync |
| **B — decode** | 1 req, long output | KV cache, quantization |
| **C — throughput** | high concurrency (64/128/256) | paged KV-cache, continuous batching |
| **D — multimodal (robot)** | **short video clip** in, short out | ViT encode, **EVS pruning** |

Report **median + p95** over ≥N repeats after warmup (exclude init/compile). OP-D uses a small fixed set of short clips (video stresses EVS hardest) — no dataset download.

**Drivers (ready-made):** vLLM `benchmark_serving` / `vllm bench serve`; **NVIDIA GenAI-Perf**; `hw2 time_generation` for eager variants.

**Quality guard (occasional, not the loop):** lossless techniques → exact-match / tight numerical-equivalence; lossy (FP8/EVS/Cache-DiT) → a tiny fixed slice (VANTAGE-Bench prompts / Physics-IQ clips) checked once per variant for acceptable drift.

**Net:** the Reasoner reproduction (≈8 techniques × 4 OPs × N) runs in **minutes on one GPU** — fast, and accurate because shapes are fixed and all four regimes are covered.

---

## 6. Technique ladders (define both the toggles and the canonical waterfall order)

Cumulative order for the waterfall; the `optimize` interface still allows any subset.

### Reasoner ladder (Part 1a) — 1× H200
| # | + technique | Path / toggle | OP |
|---|---|---|---|
| R0 | naïve HF eager, **no KV cache**, sync/step | eager (`hw2 slow_loop`) | A/B |
| R1 | + `inference_mode` | eager | A/B |
| R2 | + **KV cache** | eager | B |
| R3 | + **deferred sampling sync** | eager | A/B |
| R4 | + **torch.compile / CUDA graphs** | eager / vLLM (no `--enforce-eager`) | A |
| R5 | + **FlashAttention / fused attention** | backend flag | A/B |
| R6 | + **vLLM paged KV-cache + continuous batching** | vLLM (architectural) | C |
| R7 | + **FP8 / NVFP4 quant** *(lossy→guard)* | vLLM `--quantization` | B/C |
| R8 | + **EVS token pruning** *(lossy→guard)* | vLLM-Omni / Cosmos flag | D |

*R0–R5 cleanest in the readable eager path; R6+ in the vLLM engine (architectural ones attributed engine-vs-eager). Waterfall stitches both.*

### Generator ladder (Part 1b) — 2× H200
| # | + technique | Notes |
|---|---|---|
| G0 | naïve PyTorch reference | recompute conditioning each step |
| G1 | + **reasoner-tower output caching** | conditioning once |
| G2 | + **torch.compile / CUDA graphs** | host overhead |
| G3 | + **CFG baseline** (2 sequential forwards, 1 GPU) | the CFG cost |
| G4 | + **Cache-DiT** *(lossy→guard)* | skip redundant block compute |
| G5 | + **FP8 quant** *(lossy→guard)* | memory-bound denoise |
| G6 | + **VAE-Patch-Parallel** | shrinks decode tail |
| G7 | + **CFG-Parallel (2 GPU)** — cond on GPU0, uncond on GPU1, 1 P2P/step | **NVIDIA's technique; end of Part 1.** Note: G7's bar reflects adding a 2nd GPU (a scaling technique), label accordingly. |
| **P2** | **+ 1–2 NEW techniques (Part 2)** | picked from the breakdown; lossless preferred (§14) |

*Excluded: Ulysses CP, HSDP, CPU offload.*

---

## 7. Framework & infrastructure

| Concern | Choice | Rationale |
|---|---|---|
| **Reasoner serving** | **vLLM** (Qwen3-VL path), 1× H200 | Paged KV-cache, continuous batching, fused attention out of the box. |
| **Generator serving** | **vLLM-Omni**, 2× H200 | NVIDIA's Generator serving approach; 2 GPUs enable CFG-Parallel. |
| **Readable ablation / build techniques** | **cosmos-framework PyTorch reference path** | Report §5.3.1: primary target for new features, validated first here. Eager, per-stage timers. |
| **Latency benchmark driver** | vLLM bench / **GenAI-Perf**; `hw2 time_generation` for eager | Standard, fast, reproducible (§5). |
| **Deployment / orchestration + `optimize` slot** | Nebius `npa workbench cosmos` + MLflow | Ops shell; we add engine optimization behind its `optimize` command. |
| **Instrumentation** | Adapted from `gpu_and_inference_hw` (§9) | Timing, profiling, ablation, roofline, plots. |

### Workbench-compatibility — build into the `npa optimize` slot
`optimize_cmd` is a reserved no-op placeholder (`typer.echo("not yet implemented")`), intended for "TensorRT compilation and quantization." We implement it. Mirror `npa/src/npa/{cli,workbench}/cosmos/`:
- **CLI:** Typer `@app.command("optimize")` `optimize_cmd`, re-exported via `make_cli_wrapper("npa.cli.cosmos", "optimize_cmd", …)`. Options: `--tower {reasoner,generator}`, `--preset {none,full}`, `--enable <csv>`, `--ablate`, plus NVIDIA-style `--model`, `--backend`, `--no-guardrails`, `--output {text,json}`. Reuse `_get_config()`, `_output()`, `_fail()`, `Cosmos3ServeConfig`, `build_cosmos3_inference_args`.
- **Backend:** expose the optimized engine as a `Backend` enum value (alongside `basic|nim|triton`) so `serve`/`deploy`/`autoscale`/`status` consume it unchanged.
- **Workflow twin:** `npa.workflow/v0.0.1` YAML (`toolRef: workbench.cosmos3.optimize`, `resources.gpu.accelerators: H200:{1|2}`, `outputs` schema `npa.workbench.cosmos3.optimize.v1`) + `skypilotTwin`.
- **Artifacts:** emit waterfall / breakdown JSON to `s3://.../{{run.id}}/optimize/…` with schema versioning.
- **Module layout:** mirror `workbench.cosmos3.optimize` so upstreaming is a move, not a rewrite.

---

## 8. Requirements

### Functional
- **F1** Deploy Nano: Reasoner via vLLM (1× H200), Generator via vLLM-Omni (2× H200), + cosmos-framework eager path for readable ablation.
- **F2** Fixed, reproducible latency benchmark (§5): synthetic fixed-shape workload, 4 OPs, warmup excluding init/compile, median + p95, MLflow.
- **F3** **Selectable optimization** (§5): per-technique toggles + `--preset full|none` + `--ablate`, both towers.
- **F4** Cumulative ablation → "vs V0 / vs prev" table → contribution waterfall.
- **F5** Per-stage instrumentation (§4) + kernel trace export (torch.profiler → Perfetto; Nsight); breakdown reconciles ≤5%.
- **F6** Roofline classification of dominant stages (compute- vs memory-bound).
- **F7** Workbench-compatible `optimize` command + workflow twin (§7).
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
7. Wire the Reasoner path into the workbench-compatible `optimize` command (§7).

### Part 1b — Generator (next), 2× H200
8. Deploy Generator via vLLM-Omni + eager path; reproduce NVIDIA 256p/480p/720p numbers.
9. Run the Generator ladder G0→G7 (incl. **CFG-Parallel on 2 GPUs**); build waterfall + breakdown; quality-guard Cache-DiT/FP8; extend `optimize` to `--tower generator`.

### Part 2 — Improve (later)
10. Pick **1–2 new techniques** for the dominant stage (candidates §14); implement in the eager path first, then vLLM-Omni.
11. Prove lossless (F8); append to the same waterfall; report marginal contribution + E2E delta.
12. Ship via `optimize` + workflow twin; capture Grafana/MLflow evidence.

---

## 12. Success criteria
- **Part 1:** a workbench-compatible `optimize` command selecting any technique subset or `--preset full` for both towers; contribution waterfalls attributing each NVIDIA technique's saving across the four regimes, reconciling ≤5% to wall-clock; end-to-end within a documented margin of NVIDIA's numbers.
- **Part 2:** 1–2 new techniques with a measurable marginal contribution on the same waterfall; lossless proven (or drift reported for a guarded lossy one); no regression on the quality guard.

---

## 13. Decisions (resolved)
- Model **Cosmos 3 Nano**; **Reasoner 1× H200**, **Generator 2× H200**.
- OP-D input = **short video clip**; **vLLM** (TensorRT-LLM deferred).
- **Part 1 = reproduce all NVIDIA techniques** behind a **selectable, workbench-compatible `optimize`** (per-technique or full package); CFG-Parallel included on 2 GPUs.
- **Part 2 = 1–2 new techniques.** *(Which ones: picked from the Part-1 breakdown — see §14.)*
- `npa optimize` is a placeholder we implement (§7).

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
  optimize/          # engine techniques + toggle registry, shaped to upstream into npa.workbench.cosmos3.optimize
    techniques/      # one module per toggle (kv_cache, cuda_graphs, cfg_parallel, cache_dit, fp8, vae_patch, evs, …)
    registry.py      # technique registry + presets (none/full) + --ablate order
    cli.py           # optimize_cmd (Typer, npa semantics) — mirrors npa.cli.cosmos
  workflows/
    cosmos3-optimize.yaml   # npa.workflow/v0.0.1 twin (toolRef: workbench.cosmos3.optimize)
  observability/     # Prometheus/Grafana/DCGM + dashboards
  results/           # traces, PNGs, JSON, MLflow artifacts
```
