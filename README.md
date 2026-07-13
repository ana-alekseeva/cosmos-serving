# cosmos-serving

Reproduce and attribute NVIDIA's **Cosmos 3 serving optimizations** (tech report
§5.3), and show how each technique moves **latency** and **throughput** across
different input/output shapes. See [specification.md](specification.md) for the plan.

## What the report actually quantifies (and what this repo measures)

§5.3 puts the numbers on the **Generator** (diffusion). The **Reasoner** is "reuse
Qwen3-VL in vLLM / TensorRT-LLM out of the box" — the report gives it *no* technique
ablation. So the two towers get two different experiments:

| Tower | Experiment | Reproduces |
|---|---|---|
| **Generator** | per-clip **latency waterfall** (T2I / T2V / I2V × 256p/480p/720p) | CUDA graphs **30–60% on T2I** (§5.3.1); reasoner-cache, Cache-DiT, FP8, VAE-patch, CFG-/Context-Parallel "nearly halves" (§5.3.1/3) |
| **Generator** | **batching throughput** sweep | **Table 9** (T2V 256p 8–55%, 480p 1–5%, 720p none) |
| **Reasoner** | stock-vLLM **concurrency/shape sweep** — TTFT · latency · tok/s · req/s vs concurrency 1/64/128/256 | **1:1 with `inference_benchmarks.md`**: input=50, output {1,100}; video 1/2 FPS reproduced, text+image added |

## Run it (mock backend — no GPU)

The harness runs end-to-end on the **mock backend** (a modeled-latency table anchored
to the report's stated numbers) so the plumbing and figures are validated before the
H200. Swap `--backend mock` → `--backend vllm` on the GPU.

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Generator: latency waterfall + Table 9 batching throughput (+ JSON):
uv run python -m optimize.cli --tower generator --ablate --out-dir results
#   -> results/generator_waterfall.png, results/generator_batching.png, *.json

# Reasoner: stock-vLLM concurrency/shape sweep, 1:1 with inference_benchmarks.md
# (input=50, output {1,100}, modalities text/image/video-1fps/video-2fps x concurrency 1/64/128/256):
uv run python -m optimize.cli --tower reasoner --out-dir results
#   -> results/reasoner_sweep.png, results/reasoner_sweep.json
# The full sweep is 32 points; --op is a substring filter to scope a GPU run:
uv run python -m optimize.cli --tower reasoner --op vid --out-dir results   # video shapes only
uv run python -m optimize.cli --tower reasoner --op o100 --out-dir results  # output-100 points only

# Measure one hand-picked Generator technique subset (or --preset full):
uv run python -m optimize.cli --tower generator --enable reasoner-cache,cuda-graphs,fp8,cfg-parallel
```

## Layout

| Path | Role |
|---|---|
| `optimize/registry.py` | Generator technique toggles, presets, latency-ladder order, mock model |
| `optimize/cli.py` | `optimize` command — generator `--ablate` / subset; reasoner sweep |
| `optimize/techniques/` | real-backend wiring, one module per toggle (stubs) |
| `bench/workload.py` | Generator OP matrix + reasoner concurrency-sweep OPs + Table 9 |
| `bench/ablation.py` | cumulative `--ablate` runner + "vs V0 / vs prev" table (Generator) |
| `bench/sweep.py` | reasoner concurrency sweep + generator batching-throughput sweep |
| `bench/drivers.py` | MockEngine (runs anywhere) + VLLMEngine (H200) |
| `bench/serving.py` | vLLM / vLLM-Omni server launch + flag mapping |
| `bench/aiperf.py` | AIPerf (reasoner TTFT/throughput) + timed generation (generator) |
| `bench/plots.py` | waterfall + reasoner-sweep + batching-throughput figures |

Real H200 numbers from the previous (abandoned) reasoner-ablation framing are kept
under `results/_archive_old_framing/` for reference.

## Real backend (on the H200)

`--backend vllm` launches a vLLM (Reasoner) / vLLM-Omni (Generator) server, measures
each point via **AIPerf** (reasoner: TTFT + throughput at each concurrency) or a timed
generation request (generator), then tears the server down.

```bash
bash deploy/setup_gpu.sh          # deps + weights access (one-time)
uv run python -m optimize.cli --tower generator --ablate --backend vllm --out-dir results
uv run python -m optimize.cli --tower reasoner --backend vllm --out-dir results
```

**Before trusting the numbers**, confirm every `# VERIFY` marker in `bench/serving.py`
and `bench/aiperf.py` against your installed vLLM / vLLM-Omni / AIPerf versions (CLI
flag names, generation endpoint/payload, AIPerf JSON schema, multimodal input). These
were written from docs, not run on a GPU. Architectural vLLM features (paged attention,
continuous batching, fused attention) are always-on — the reasoner sweep characterizes
them as the stock config, it does not toggle them.

## Deploy Cosmos on Nebius with the workbench (`npa`)

An alternative to `deploy/setup_gpu.sh`: use Nebius's
[`nebius-physical-ai`](https://github.com/nebius/nebius-physical-ai) workbench to stand
up a managed Cosmos serving endpoint. This is the fastest way to *run* Cosmos; it is
**separate** from the ablation harness above, which manages its own per-variant servers
for measurement.

```bash
uv tool install "git+https://github.com/nebius/nebius-physical-ai.git#subdirectory=npa"
npa configure --interactive      # Nebius profile: tenant/project/region/bucket
export HF_TOKEN=hf_...            # gated Cosmos weights (accept the license on HF first)

npa workbench cosmos -p <project-alias> -n cosmos deploy \
  --runtime serverless --gpu-type gpu-h200-sxm --gpu-preset <preset> --wait
npa workbench cosmos -p <project-alias> -n cosmos serve         # deploy leaves it UNLOADED
npa workbench cosmos -p <project-alias> -n cosmos infer \
  --prompt "A robot arm stacks colored cubes on a table" \
  --output-path s3://<your-bucket>/cosmos/out/ --output-format json
npa workbench cosmos -p <project-alias> -n cosmos teardown --yes   # stop billing
```

Note: the workbench's own `npa … cosmos optimize` is a roadmap placeholder (`not yet
implemented`) — the optimization work in this repo is what fills that slot.
