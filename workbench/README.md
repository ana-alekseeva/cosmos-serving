# workbench/ — provision the Nebius infrastructure (serverless jobs + optimized endpoint)

The npa-workbench (`nebius/nebius-physical-ai`) integration, used **only to provision
infrastructure** on npa-managed Nebius infra (no resources created by hand):

1. **serverless GPU jobs** that run the repo's ablation and evaluation code on managed GPUs
   (see [../jobs/](../jobs/)), and
2. **a serverless endpoint** serving `Cosmos3-Nano-Policy-DROID` with the ablation-winning
   latency optimizations baked in.

**The optimization logic itself lives in the repository**, not here — the configuration matrix,
the ablation runner, the waterfalls, and the RoboLab evaluation are all in `policy/`,
`run_matrix.py`, `aggregate.py`, and `run_robolab.py`. This directory is the canonical, versioned
record of *what infra to stand up*.

| File | Role |
|---|---|
| [policy-droid.config.yaml](policy-droid.config.yaml) | serve `Cosmos3-Nano-Policy-DROID` serverless with the final E4 optimizations |
| [optimized-deploy.config.yaml](optimized-deploy.config.yaml) | deploy config (equivalent knobs; used by `../jobs/deploy-optimized.sh`) |
| [../jobs/](../jobs/) | the executable Nebius job specs (SkyPilot + npa.workflow twins + deploy script) |

Bucket: **`s3://serverless-challenge/cosmos3-ablation-results/`**.

## One-time: install + configure npa

npa installs from a checkout (Python 3.10+); there is no `uv tool install` / `pipx` path:

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai && python3 -m venv .venv && . .venv/bin/activate && pip install -e npa
npa configure --interactive        # ~/.npa/{credentials,config}.yaml: project/tenant/region,
                                   # S3 bucket + key, HF_TOKEN. Needs the AI Jobs IAM role.
```

## Create the serverless endpoint (the workbench resource)

`npa workbench cosmos deploy --runtime serverless` creates a Nebius Serverless AI Endpoint for
the Cosmos container (verified against npa 0.1.0). The repository-level
[README](../README.md) is the full runbook; the deploy script wraps the real command:

```bash
MODE=optimized PROJECT_ALIAS=cosmos HF_TOKEN=$HF_TOKEN bash ../jobs/deploy-optimized.sh
# then measure it with the repo harness:
python ../run_matrix.py --backend vllm --endpoint https://<endpoint-url> --configurations E4
```
(The `*.sky.yaml` specs are optional raw-SkyPilot specs for RoboLab on RT-core GPUs, not npa.)

## Deploy the optimized-model endpoint (serverless)

After the E0-E4 latency run, evaluate E4 with the RoboLab subset. Accept it only when
`results/aggregate/robolab_subset.json` reports a passing gate, then deploy the retained endpoint:

```bash
HF_TOKEN=$HF_TOKEN GPU_TYPE=gpu-h100-sxm GPU_PRESET=1gpu-16vcpu-200gb \
  bash ../jobs/deploy-optimized.sh
# ... point Job 3 (RoboLab subset) at the endpoint, then stop billing:
npa workbench cosmos -p serverless-challenge -n cosmos-policy teardown --yes
```

## `# VERIFY`
- `--gpu-type` / `--gpu-preset` strings (`nebius compute platform list`).
- how npa forwards engine args to the served engine (`--extra-serve-args` vs a served config;
  `npa workbench cosmos deploy --help`).
- the DROID policy **infer payload** (2 camera views + instruction + 8-D proprio → 32×8 chunk).
- engine flags confirmed against the vLLM-Omni serve CLI reference (same flags
  `policy/serving.py::engine_args()` maps).
