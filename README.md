# cosmos-serving — Cosmos3-Nano-Policy-DROID latency attribution & optimization

Attribute and reduce the inference latency of **`Cosmos3-Nano-Policy-DROID`** on the *only*
evaluated task ([specification_revised.txt](specification_revised.txt) §1):

```text
DROID camera observations + language instruction + proprioceptive state
    → 32 × 8 robot-action chunk
```

The model is evaluated by executing its generated actions in **RoboLab**. This repo runs the
single-GPU PyTorch ablation matrix, attributes each optimization to the pipeline stage it
shrinks, validates the winner through the production vLLM / vLLM-Omni engines, and gates the
lossy techniques on RoboLab task success.

## Two waterfalls (specification_revised.txt §3)

| Waterfall | Rungs | Measures |
|---|---|---|
| **Native PyTorch (P)** | P0 cuDNN-fused → +`torch.compile` → +CUDA graphs → **+conditioning cache** | `total_chunk_ms` (+ per-stage breakdown) |
| **End-to-end (E, vLLM)** | E0 eager (math) → +Flash → +compile → +CUDA graphs → +reasoner cache → +Cache-DiT → +FP8 → **final** | `total_chunk_ms` |

The old reasoner (R) and generator (G) ladders are merged into one native ladder **P** — they ran the
same single MoT inference and gave identical on-box numbers. Cache-DiT/FP8 are vLLM-only (§5.3.3), on E.

Cache-DiT and FP8 are **lossy → quality-gated**: included in the final configuration only if
RoboLab success holds (§3, §9). The multi-GPU strategies (CFG-Parallel, Ulysses Context-Parallel)
run as a **separate** experiment (§3) — never mixed into the single-GPU waterfall.

## Run it (mock backend — no GPU)

The harness runs end-to-end on the **mock backend** (a modeled per-stage latency table anchored
to the spec's §7 example log) so the plumbing, per-request JSONL logs, waterfalls, stage
breakdown, and aggregation are validated before the GPU. Swap `--backend mock` → `--backend vllm`
on the target inference GPU.

Managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Job 1 — the full single-GPU ablation matrix (P0-P3, E0-E6) with §8 bias controls
# (baseline at start+end, randomized order, drift rejection), each config an isolated subprocess:
uv run python run_matrix.py --config config/experiment.yaml \
    --input-manifest policy/mock/manifest.json --output-dir results --backend mock

# Job 5 — aggregate: waterfalls + stage breakdown + CIs + CSV/Parquet + quality tables + figures:
uv run python aggregate.py --out-dir results
#   -> results/aggregate/waterfall_{native,end_to_end}.png
#      results/aggregate/stage_breakdown.png, quality_comparison.png, summary.csv

# One configuration on its own (the §7 per-config artifacts):
uv run python run_configuration.py --configuration P0 --backend mock --out-dir results

# RoboLab subset quality gate (baseline vs final) — the lossy Cache-DiT/FP8 gate:
uv run python run_robolab.py --baseline E0 --candidate E6

# Separate multi-GPU experiment (CFG-Parallel + Ulysses vs best single-GPU):
uv run python run_multigpu.py --backend mock
```

Example end-to-end result (mock, 4-step DROID recipe): **E0 ≈ 229 ms → E6 ≈ 57 ms (4.0×)**,
dominated by the reasoner-conditioning cache (the naive baseline recomputes conditioning every
one of the 4 denoising steps: reasoner P0 183 ms → P3 28 ms); Cache-DiT + FP8 pass the RoboLab
subset gate. See `results/aggregate/`.

## Layout

| Path | Role |
|---|---|
| `run_matrix.py` | Job 1 — ablation-matrix orchestrator (subprocess per config, §8 bias controls) |
| `run_configuration.py` | single configuration → the five §7 log artifacts |
| `aggregate.py` | Job 5 — merge logs → CSV/Parquet + waterfalls + stage breakdown + CIs + figures |
| `run_robolab.py` / `run_multigpu.py` | Jobs 3-4 quality gate / the separate multi-GPU experiment |
| `config/experiment.yaml` | the experiment config (§4 `--config experiment.yaml`) |
| `policy/configs.py` | the P0-P3 / E0-E6 configuration matrix + stage-effect model |
| `policy/dataset.py` | fixed ~256-request replay set + the 18-task RoboLab quality subset (§5) |
| `policy/pipeline.py` | mock per-stage latency engine + vLLM/vLLM-Omni real-backend stub |
| `policy/measure.py` / `logs.py` | the §6 latency field set, p50/p90/p99, §7 log format |
| `policy/matrix.py` / `runner.py` | matrix orchestration core / single-config runner core |
| `policy/aggregate.py` / `plots.py` | aggregation + the waterfall / stage-breakdown / quality figures |
| `policy/serving.py` | real-backend engine-flag mapping + the §9 forced-SDPA / static-shape rules |
| `policy/robolab.py` / `multigpu.py` | RoboLab task-success gate / CFG-/Ulysses-parallel experiment |
| `jobs/` | the five-job Nebius plan (§4) + optimized-endpoint deploy |
| `workbench/` | Nebius infra-provisioning configs (serve the optimized policy endpoint) |
| `results/` | waterfalls, stage breakdown, quality tables, CSV, one example per-config log dir |

## Latency measurement (specification_revised.txt §6)

Every request records the full field set — `preprocess_ms h2d_ms reasoner_ms
generator_prepare_ms denoising_ms denoising_step_ms[] postprocess_ms d2h_ms server_ms
transport_ms first_action_ms total_chunk_ms peak_memory_mb` — with CUDA events for GPU stages
and monotonic timers end-to-end, batch size 1, ~25 warm-ups (excluded), ≥200 measured requests,
and p50/p90/p99 summaries. One JSONL row per request (§7) plus `summary.json`,
`environment.json`, `system-info.json`, `status.json` per configuration.

## Real backend (on the GPU)

`--backend vllm` launches vLLM (Reasoner) / vLLM-Omni (Generator / full policy) with each
configuration's engine flags and measures wall-clock from the server's per-stage timers.

```bash
bash deploy/setup.sh                                        # deps + weights (one-time); prints run cmds
uv run python -m policy.capture --n 50 --out /local/replay  # capture the real DROID replay set
uv run python run_matrix.py --input-manifest /local/replay/manifest.json \
    --output-dir results --backend pytorch                 # routes P->pytorch, E->vLLM-Omni
uv run python aggregate.py --out-dir results
```

**Before trusting the numbers**, confirm every `# VERIFY` in `policy/serving.py` against your
installed vLLM / vLLM-Omni (engine flag names, the forced-SDPA Flash backend that must *fail*
rather than silently fall back, static/bucketed shapes for the CUDA-graph configs, and the
per-stage timing response). The mock stands in for all of this offline.

## Run the ablation jobs on Nebius (`nebius ai job create`)

The jobs run as containers. Build the **native (cu130) image** ([deploy/Dockerfile](deploy/Dockerfile)) —
it bakes torch-cu130 + `cosmos_framework` + this harness and bumps cuDNN to ≥ 9.22 for the fused-attention
baseline. cosmos-framework's `vllm` and `cu130` uv groups **conflict**, so this image is the native
`P0–P3` runtime; the vLLM `E`-ladder + jobs 2/2b need a **separate `--group vllm` image**. All indexes are
public — no build secret needed.

Push it to a Nebius Container Registry:
```bash
PROJECT_ID=<project-id>; REGION=eu-north1
nebius registry create --name cosmos-droid --parent-id "$PROJECT_ID"    # once; note the registry-… id
REGISTRY_ID=<registry-id>                                               # nebius registry list --parent-id "$PROJECT_ID"
nebius iam get-access-token | docker login "cr.${REGION}.nebius.cloud" --username iam --password-stdin
IMAGE="cr.${REGION}.nebius.cloud/${REGISTRY_ID}/cosmos-droid-bench-native:latest"
docker build --platform linux/amd64 -f deploy/Dockerfile -t "$IMAGE" . # x86_64 wheels — build on x86 for speed
docker push "$IMAGE"
```

Launch a job — [deploy/run_job.sh](deploy/run_job.sh) is the env-driven entrypoint (stages replay + model,
runs the matrix, aggregates, optional S3 upload). Load secrets from `.env` first so the `$VARS` expand:
```bash
set -a && source .env && set +a          # HF_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
nebius ai job create --name cosmos-job1-native --parent-id "$PROJECT_ID" \
  --image "$IMAGE" --platform gpu-h200-sxm --preset 1gpu-16vcpu-200gb --shm-size 64Gi \
  --inject-file deploy/run_job.sh:/run_job.sh --container-command bash --args /run_job.sh \
  --env BACKEND=pytorch --env MODE=matrix --env CONFIGS=P0,P1,P2,P3 \
  --env REPLAY_SIZE=50 --env WARMUPS=5 --env OUTPUT_DIR=results \
  --env HF_TOKEN="$HF_TOKEN"
```
`--timeout` is a *cap* (default 24h) — you're billed for actual runtime (~1–2 h for Job 1). The other jobs are
the same shape with different `--preset`/`--env` (and the vLLM image): **1b** `MODE=multigpu` on `2gpu-…`;
**2** `BACKEND=vllm CONFIGS=E0,E6`; **2b** adds `TENSOR_PARALLEL_SIZE=2 PARALLEL=cfg` on 2 GPUs.

## Create Nebius resources with the workbench (`npa`)

Provision on Nebius with the **Nebius Physical AI workbench** (`npa`), installed into this
project's venv. Commands below are the real, current CLI (npa 0.1.0), verified on-box.

```bash
# One-time: install npa editable into this project, then configure (needs the "AI Jobs" IAM role)
bash deploy/install_npa.sh
npa configure --interactive

# Create a serverless AI endpoint for the optimized (E6) DROID policy — the workbench resource:
MODE=optimized PROJECT_ALIAS=cosmos HF_TOKEN=$HF_TOKEN bash jobs/deploy-optimized.sh
#   -> npa workbench cosmos deploy --runtime serverless --model nvidia/Cosmos3-Nano-Policy-DROID
#      --gpu-type gpu-h200-sxm --gpu-preset 1gpu-16vcpu-200gb --env ... --auth token --wait

# Measure / gate the deployed endpoint with the repo harness:
python run_matrix.py --backend vllm --endpoint https://<endpoint-url> --configurations E6
```

Full runbook (baseline vs optimized, RoboLab gate, teardown, and the verified-flags reference)
in [jobs/README.md](jobs/README.md). The ablation/eval logic stays **in this repo**; the
workbench only provisions the infra — its own `npa … cosmos optimize` is a `Roadmap placeholder`,
which we don't depend on.
