# cosmos-serving

Reproduce NVIDIA's Cosmos 3 serving optimizations on Nebius, attribute each
technique's latency contribution, and (Part 2) add one or two new techniques.
See [specification.md](specification.md) for the full plan.

## What runs today (Part 1a skeleton, mock backend — no GPU)

The harness runs end-to-end on the **mock backend**, producing the real
contribution waterfall from a modeled latency table so the plumbing is validated
before touching the H200. Swap `--backend mock` → `--backend vllm` on the GPU.

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # create the env from pyproject.toml + uv.lock

# Reasoner technique-contribution waterfall (4 operating points) + JSON:
uv run python -m optimize.cli --tower reasoner --ablate --out-dir results
#   -> results/reasoner_waterfall.png, results/reasoner_ablation.json

# Pick a subset, or the full NVIDIA package:
uv run python -m optimize.cli --tower reasoner --enable kv-cache,cuda-graphs,fp8,evs --output json
uv run python -m optimize.cli --tower generator --preset full

# Stdlib-only quick table:
uv run python -m bench.ablation reasoner
```

Deps live in `pyproject.toml` (`uv add <pkg>` to add). On the H200, install the
real backend with `uv sync --extra gpu` (plus torch / vllm / vllm-omni per the
GPU environment).

## Layout

| Path | Role |
|---|---|
| `optimize/registry.py` | technique toggles, presets, ablation order, mock model |
| `optimize/cli.py` | `optimize` command (Typer, npa semantics) — subset / `--preset full` / `--ablate` |
| `optimize/techniques/` | real-backend wiring, one module per toggle (stubs) |
| `bench/workload.py` | operating-point matrix A–D (OP-D = short video clip) |
| `bench/drivers.py` | MockEngine (runs anywhere) + VLLMEngine (H200 stub) + `time_generation` |
| `bench/ablation.py` | cumulative `--ablate` runner + "vs V0 / vs prev" table |
| `bench/plots.py` | `plot_contribution_waterfall()` + `plot_stage_breakdown()` |
| `bench/stages.py` | per-stage CUDA-event timers + `reconcile()` (≤5%) |
| `bench/roofline.py` | H200/H100 GPU specs + compute/memory-bound classification |
| `bench/equivalence.py` | Part-2 lossless checks (token/latent/SSIM) |
| `workflows/cosmos3-optimize.yaml` | `npa.workflow/v0.0.1` twin (`toolRef: workbench.cosmos3.optimize`) |

## Next (on the H200)

Wire `VLLMEngine.measure` and `optimize/techniques/*.apply` to launch vLLM /
vLLM-Omni with each toggle's `engine_flags` and measure via GenAI-Perf / vLLM
bench. The registry, ablation, plots, and CLI stay unchanged.
