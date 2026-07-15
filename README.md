# cosmos-serving — Cosmos3-Nano-Policy-DROID latency validation

**A robot cannot wait for the next token.** To place an object on a shelf, a policy must
combine two exterior-camera views, a wrist-camera view, a language instruction, and the
robot's state, then return the next actions quickly enough to stay inside the control loop.
Cosmos 3 brings that reasoning and action generation into one model, but its multimodal
Reasoner and diffusion-based Generator make serving substantially more involved than
serving a conventional LLM.

<table>
  <tr>
    <td width="33%"><img src="docs/assets/droid_exterior_1.png" alt="DROID exterior camera 1 showing a Franka robot arm, a yellow cup, and the target object."></td>
    <td width="33%"><img src="docs/assets/droid_exterior_2.png" alt="DROID exterior camera 2 showing the robot workspace from a second angle."></td>
    <td width="33%"><img src="docs/assets/droid_wrist.png" alt="DROID wrist camera showing the gripper approaching the target object."></td>
  </tr>
  <tr>
    <td align="center"><sub>Exterior camera 1</sub></td>
    <td align="center"><sub>Exterior camera 2</sub></td>
    <td align="center"><sub>Wrist camera</sub></td>
  </tr>
</table>

This project turns that question into a practical workflow: **how much latency can we
remove without making the robot worse?** The repository already contains the benchmark
matrix, cloud-job definitions, serving configuration, trace analysis, and plotting code.
You can check the complete pipeline locally without a GPU, then launch the real benchmark
as one cloud job when you are ready.

There are two ready-to-run cloud workflows:

| Job | File | What it does |
|---|---|---|
| **Job 2** | [jobs/job2-production-validation.sky.yaml](jobs/job2-production-validation.sky.yaml) | Runs the complete **E0–E4** latency ladder through vLLM / vLLM-Omni on one H100, then uploads raw per-configuration results and profiler traces to object storage. |
| **Job 3** | [jobs/job3-robolab-subset.sky.yaml](jobs/job3-robolab-subset.sky.yaml) | Runs the 18-task RoboLab subset against a baseline and an optimized endpoint, uploads success records, and gates FP8 on task success. |

The optimization ladder (`E0..E4`) is defined in [policy/configs.py](policy/configs.py):
`E0` eager (SDPA) → `E1` +Flash Attention → `E2` +torch.compile → `E3` +CUDA graphs → `E4` +FP8 (lossy).

**Most users can start with the local smoke test and Job 2.** Deploy Job 3 only when you
also want to verify that the faster FP8 policy preserves task success in RoboLab.

## Start here: try the pipeline locally

No GPU or cloud account is needed for this first run. The mock backend exercises the same
benchmark, logging, aggregation, and plotting flow without loading the model:

```bash
uv sync
uv run python run_matrix.py --smoke --input-manifest policy/mock/manifest.json \
  --output-dir results-smoke --backend mock \
  --configurations E0,E1,E2,E3,E4
uv run python aggregate.py --out-dir results-smoke
```

The generated figures and summaries appear in `results-smoke/aggregate/`. To check the
repository itself as well, run `uv run --group dev pytest -q`.

---

## Prepare for a cloud run (once)

If the local flow looks good, add the credentials used to download the model and upload
results. Dependency setup is already handled by [uv](https://docs.astral.sh/uv/):

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

Cloud jobs use [SkyPilot](https://docs.skypilot.co/) (`sky`) with a Kubernetes-backed
Nebius cluster. Before the first run, fill in the project-specific values marked `# VERIFY`
in the job YAMLs, such as the image tag, repository URL, and endpoint URLs.

---

## Run the latency benchmark

Once the credentials are in place, one command launches the complete E0–E4 experiment on
an H100:

```bash
set -a && source .env && set +a
sky jobs launch jobs/job2-production-validation.sky.yaml --secret HF_TOKEN \
  --env CONFIGS=E0,E1,E2,E3,E4
```

The job captures the real DROID replay set, runs all five cumulative configurations with
`run_matrix.py --backend vllm`, aggregates the measurements, and uploads them to
`s3://serverless-challenge/cosmos3-ablation-results/production/`. Raw per-configuration
logs are stored under `production/raw/`; vLLM profiler traces are stored as an extracted
directory tree under `production/raw/traces/`.

For a cheaper baseline-versus-final validation after the complete ladder has already been
measured, override the same job with `--env CONFIGS=E0,E4`. That two-configuration run cannot
reproduce the intermediate E1–E3 bars.

## Optional: deploy the E0 and E4 policy endpoints

This section is only needed for the RoboLab quality gate; skip it if you only need the
latency benchmark and traces. Job 3 scores two **already-running** endpoints—an **E0**
baseline (eager) and the **E4** optimized candidate (FP8). Both serve
`Cosmos3-Nano-Policy-DROID` through the *same* vLLM-Omni stack Job 2 uses, and each process
exposes **both** eval routes: `POST /v1/videos` (the latency harness) and
`ws /v1/realtime/robot/openpi` (RoboLab).

### Install and configure npa

```bash
bash deploy/install_npa.sh            # install npa (editable) into the project venv
npa configure --interactive           # writes ~/.npa/{credentials,config}.yaml — needs the AI Jobs IAM role
```

### Build and push the serving image

`npa cosmos deploy --runtime serverless` runs the image's **own ENTRYPOINT** (it injects env
vars, never a container command), so the serving image must start the server itself.
[deploy/Dockerfile.serve](deploy/Dockerfile.serve) is a thin layer on the Job-2 benchmark image
`cosmos-droid-vllm:v3` that adds exactly that entrypoint — one image serves **either** rung, with
the `CONFIG` env var selecting E0 vs E4. Bump the tag on every push (a reused tag is served stale
from the k8s node cache):

```bash
REG=cr.eu-north1.nebius.cloud/e00k6drmprp0pm6zcf
docker build --platform linux/amd64 -f deploy/Dockerfile.serve -t $REG/cosmos-droid-vllm:v4 .
docker push $REG/cosmos-droid-vllm:v4
```

The entrypoint ([deploy/serve_endpoint.sh](deploy/serve_endpoint.sh) →
[deploy/serve_launch.py](deploy/serve_launch.py)) builds the exact serve command from
`policy/serving.py::engine_args(CONFIG)` — so E0 forces `TORCH_SDPA` + eager and E4 is the full
cumulative `FLASH_ATTN` + `VLLM_COMPILE`/`FULL_AND_PIECEWISE` CUDA graphs + FP8 — and appends the
`--stage-overrides` `policy_server_config` that mounts the OpenPI websocket route. There is no
separate "bake the flags into the image" step: the single `CONFIG` var drives everything.

### Deploy both endpoints on H100

```bash
export HF_TOKEN=...                    # Cosmos3-Nano-Policy-DROID license accepted on HF
MODE=baseline  bash jobs/deploy-optimized.sh    # E0 -> workbench alias cosmos-policy-baseline
MODE=optimized bash jobs/deploy-optimized.sh    # E4 -> workbench alias cosmos-policy-optimized
```

Defaults ([jobs/deploy-optimized.sh](jobs/deploy-optimized.sh)): `-p eu-north1`, `gpu-h100-sxm` /
`1gpu-16vcpu-200gb`, `--auth none` (RoboLab and the harness send no auth header), and
`IMAGE=…/cosmos-droid-vllm:v4`. `--wait` blocks until RUNNING; each deploy stores the endpoint URL
in its workbench alias. Override any default with the matching env var (`GPU_TYPE`, `IMAGE`, …).

### Check both routes

```bash
curl -sf <url>/health && echo OK
curl -si -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" <url>/v1/realtime/robot/openpi | head -1
#   HTTP/1.1 101  -> OpenPI route mounted (good)
#   404           -> stage-overrides / cosmos_framework not active; RoboLab would fail
```

Stop billing when finished:
`npa workbench cosmos -p eu-north1 -n cosmos-policy-<mode> teardown --yes`.

## Optional: run the RoboLab quality gate

With both endpoints running and the 18 task slots in
[config/robolab_tasks.yaml](config/robolab_tasks.yaml) populated (they already are — the harness
rejects missing or unknown tasks), launch one job per endpoint (they run in parallel):

```bash
sky jobs launch jobs/job3-robolab-subset.sky.yaml -c robolab-e0 \
  --env ROLE=baseline  --env COSMOS_ENDPOINT_BASELINE=https://<baseline-endpoint>
sky jobs launch jobs/job3-robolab-subset.sky.yaml -c robolab-e4 \
  --env ROLE=optimized --env COSMOS_ENDPOINT_OPTIMIZED=https://<optimized-endpoint>
```

Records upload to `s3://serverless-challenge/robolab-eval-results/subset/raw/robolab/`.

---

## Bring the results back locally

`results/` is the local download target (git-ignored except for `.gitkeep`).

```bash
set -a && source .env && set +a

# Job 2 raw results (E0-E4 request logs plus the extracted traces/ directory)
aws s3 cp s3://serverless-challenge/cosmos3-ablation-results/production/raw/ results/ \
  --recursive --endpoint-url "$AWS_ENDPOINT_URL"

# Job 3 RoboLab records (optional — for the local gate)
aws s3 sync s3://serverless-challenge/robolab-eval-results/subset/raw/robolab/ results/robolab/ \
  --endpoint-url "$AWS_ENDPOINT_URL"
```

## Create the benchmark plots

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
# Job 2 uploads a directory, not traces.tar.gz. Download it before analysis if the
# complete production/raw/ tree was not downloaded above:
aws s3 cp \
  s3://serverless-challenge/cosmos3-ablation-results/production/raw/traces/ \
  results/traces/ --recursive --endpoint-url "$AWS_ENDPOINT_URL"

uv run python analyze_traces.py --results-dir results --traces results/traces
```

To gate Job 3 (baseline vs optimized success rates) after both endpoints have run:

```bash
uv run python run_robolab.py --backend vllm --side both --robolab-root RoboLab \
  --endpoint-baseline https://<baseline> --endpoint-candidate https://<optimized>
```
