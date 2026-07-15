# cosmos-serving — Cosmos3-Nano-Policy-DROID latency validation

Validate and analyze the inference latency of **`Cosmos3-Nano-Policy-DROID`** through the
production **vLLM / vLLM-Omni** serving stack, and gate the lossy optimization (FP8) on
**RoboLab** task success.

The repo is scoped to two cloud jobs plus local analysis:

| Job | File | What it does |
|---|---|---|
| **Job 2** | [jobs/job2-production-validation.sky.yaml](jobs/job2-production-validation.sky.yaml) | Runs the production baseline **E0** and final optimized **E4 (FP8)** through vLLM / vLLM-Omni on an H200, uploads raw per-config results + traces to object storage. |
| **Job 3** | [jobs/job3-robolab-subset.sky.yaml](jobs/job3-robolab-subset.sky.yaml) | Runs the 18-task RoboLab subset against a baseline and an optimized endpoint, uploads success records, and gates FP8 on task success. |

The optimization ladder (`E0..E4`) is defined in [policy/configs.py](policy/configs.py):
`E0` eager (SDPA) → `E1` +Flash Attention → `E2` +torch.compile → `E3` +CUDA graphs → `E4` +FP8 (lossy).

Managed with [uv](https://docs.astral.sh/uv/).

---

## 0. One-time setup

```bash
uv sync                          # create the .venv and install deps
cp .env.example .env             # then fill in the secrets below
```

`.env` supplies credentials used by the download/upload steps (sourced automatically by
`jobs/aggregate-local.sh`):

```bash
HF_TOKEN=...                     # Hugging Face token (model download)
AWS_ACCESS_KEY_ID=...            # Nebius object-storage key
AWS_SECRET_ACCESS_KEY=...
AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
OUTPUT_URI=s3://serverless-challenge/cosmos3-ablation-results/
```

Cloud jobs are launched with [SkyPilot](https://docs.skypilot.co/) (`sky`) against a
Kubernetes-backed Nebius cluster. Confirm every `# VERIFY` comment in the two job YAMLs
(image tags, repo URL, endpoint URLs) before trusting the numbers.

---

## 1. Run Job 2 — production validation (E0 vs E4)

```bash
sky jobs launch jobs/job2-production-validation.sky.yaml --secret HF_TOKEN --env CONFIGS=E0,E4
```

The job captures the real DROID replay set, runs `run_matrix.py` (backend `vllm`) for `E0,E4`,
aggregates inline, and uploads to `s3://serverless-challenge/cosmos3-ablation-results/production/`
(raw per-config logs under `production/raw/`, vLLM profiler traces under `production/raw/traces/`).

## 2. Run Job 3 — RoboLab subset gate

Job 3 scores two **already-deployed** endpoints (baseline + optimized). Deploy them first with
the workbench tooling (optional — see [jobs/deploy-optimized.sh](jobs/deploy-optimized.sh) /
[deploy/install_npa.sh](deploy/install_npa.sh)), then fill the 18 task slots in
[config/robolab_tasks.yaml](config/robolab_tasks.yaml) (the run refuses to start until they are).

Launch one job per endpoint (they run in parallel):

```bash
sky jobs launch jobs/job3-robolab-subset.sky.yaml -c robolab-e0 \
  --env ROLE=baseline  --env COSMOS_ENDPOINT_BASELINE=https://<baseline-endpoint>
sky jobs launch jobs/job3-robolab-subset.sky.yaml -c robolab-e4 \
  --env ROLE=optimized --env COSMOS_ENDPOINT_OPTIMIZED=https://<optimized-endpoint>
```

Records upload to `s3://serverless-challenge/robolab-eval-results/subset/raw/robolab/`.

---

## 3. Download the results into `results/`

`results/` is the local download target (git-ignored except for `.gitkeep`).

```bash
set -a && source .env && set +a

# Job 2 raw results (per-config latency logs + traces)
aws s3 cp s3://serverless-challenge/cosmos3-ablation-results/production/raw/ results/ \
  --recursive --endpoint-url "$AWS_ENDPOINT_URL"

# Job 3 RoboLab records (optional — for the local gate)
aws s3 sync s3://serverless-challenge/robolab-eval-results/subset/raw/robolab/ results/robolab/ \
  --endpoint-url "$AWS_ENDPOINT_URL"
```

## 4. Create the Job 2 plots

**Summary figures** (waterfall, stage breakdown, quality comparison) — the primary Job 2 plots:

```bash
uv run python aggregate.py --out-dir results
#   -> results/aggregate/waterfall_end_to_end.png
#      results/aggregate/stage_breakdown.png
#      results/aggregate/quality_comparison.png   (+ summary.csv, *.json)
```

Or use the one-shot helper, which downloads the raw trees from object storage **and** aggregates:

```bash
./jobs/aggregate-local.sh                 # download + aggregate + re-upload figures
NO_UPLOAD=1 ./jobs/aggregate-local.sh     # local figures only, no upload
```

**Trace deep-dive figures** (latency ECDFs, latency-vs-VRAM Pareto, kernel/model-part
attribution) from the vLLM profiler traces:

```bash
uv run python analyze_traces.py --results-dir results --traces results/traces.tar.gz
```

To gate Job 3 (baseline vs optimized success rates) after both endpoints have run:

```bash
uv run python run_robolab.py --backend vllm --side both --robolab-root RoboLab \
  --endpoint-baseline https://<baseline> --endpoint-candidate https://<optimized>
```

---

## Repository layout

| Path | Role |
|---|---|
| `jobs/job2-*.sky.yaml`, `jobs/job3-*.sky.yaml` | The two SkyPilot cloud jobs |
| `jobs/aggregate-local.sh` | Download raw results + regenerate Job 2 figures locally |
| `jobs/deploy-optimized.sh`, `deploy/install_npa.sh`, `deploy/versions.env`, `workbench/` | Provision the baseline/optimized endpoints Job 3 scores (Nebius `npa` workbench) |
| `run_matrix.py` | Ablation-matrix orchestrator (subprocess per config); driven by Job 2 |
| `run_configuration.py` | Single configuration → per-config log artifacts (spawned by `run_matrix.py`) |
| `run_robolab.py` | RoboLab quality gate; driven by Job 3 |
| `aggregate.py` | Merge logs → CSV + waterfalls + stage breakdown + quality figures |
| `analyze_traces.py` | Trace/profiler deep-dive figures (ECDF, Pareto, kernel/model-part) |
| `config/experiment.yaml`, `config/robolab_tasks.yaml` | Experiment config + the 18 RoboLab task slots |
| `policy/` | The pipeline: config matrix, replay dataset, latency measurement, aggregation, plots, vLLM serving contract, RoboLab runner |
| `tests/` | Harness invariant tests (`uv run --group dev pytest`) |
| `results/` | Local download target for Job 2 / Job 3 outputs |

## Local smoke test (no GPU)

The pipeline runs end-to-end on the modeled **mock backend**, which validates the plumbing,
logs, aggregation, and figures before spending GPU time:

```bash
uv run python run_matrix.py --input-manifest policy/mock/manifest.json \
    --output-dir results --backend mock --configurations E0,E4
uv run python aggregate.py --out-dir results
uv run --group dev pytest -q
```
