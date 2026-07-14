# jobs/ — create Nebius resources with the workbench (npa)

Use the **Nebius Physical AI workbench** (`npa`) to provision resources on Nebius. Everything
here was verified against **npa 0.1.0** installed into this project's venv — the commands below
are the real, current CLI (not guesses); the deploy command was confirmed to reach the Nebius
VPC API on a dry-run.

## 0. Install npa into this project + configure

npa must be installed **editable from a source checkout** (the plain wheel install crashes at
startup — it reads `[tool.npa.supported-tools]` from a pyproject.toml that isn't shipped). The
helper does the durable, project-local install:

```bash
bash deploy/install_npa.sh          # clones .nebius-physical-ai/ (gitignored), uv pip install -e
npa configure --interactive         # ~/.npa/{credentials,config}.yaml: project/tenant/region,
                                    # S3 bucket + key, HF_TOKEN. Needs the "AI Jobs" IAM role.
npa workbench --help                # confirm the tool tree
```

## 1. Create the serverless endpoint (the workbench resource)

`npa workbench cosmos deploy --runtime serverless` **creates a Nebius Serverless AI Endpoint**
for the Cosmos container and stores its URL in the workbench alias — this is the workbench
provisioning the resource. [deploy-optimized.sh](deploy-optimized.sh) wraps the real command:

```bash
# Optimized (E6) endpoint — reasoner-conditioning cache + Cache-DiT + FP8 as container env vars
# (deploy has --backend + repeatable --env; there is NO --extra-serve-args).
MODE=optimized PROJECT_ALIAS=cosmos HF_TOKEN=$HF_TOKEN bash jobs/deploy-optimized.sh
# Baseline (eager) endpoint for comparison:
MODE=baseline  PROJECT_ALIAS=cosmos HF_TOKEN=$HF_TOKEN bash jobs/deploy-optimized.sh
```

The bare command it runs (verified flags — `-p/-n` are group options before the subcommand):

```bash
npa workbench cosmos -p cosmos -n cosmos-policy-optimized deploy \
  --runtime serverless --model nvidia/Cosmos3-Nano-Policy-DROID \
  --gpu-type gpu-h200-sxm --gpu-preset 1gpu-16vcpu-200gb \
  --env POLICY_CONDITIONING_CACHE=1 --env CACHE_DIT=1 --env QUANTIZATION=fp8 \
  --auth token --wait --output json
npa workbench cosmos -p cosmos -n cosmos-policy-optimized status --output json
npa workbench cosmos -p cosmos -n cosmos-policy-optimized teardown --yes   # stop billing
```

- `--auto-serve` (default) loads the model after a healthy deploy — no separate `serve` needed.
- Platform/preset: `gpu-h200-sxm` / `1gpu-16vcpu-200gb` (1 GPU); use an 8-GPU preset for CFG-Parallel.
- `--subnet-id` is required when the project has multiple subnets; `--project-id` when it isn't in config.

## 2. Measure / gate the deployed endpoint(s)

The endpoint serves the DROID action policy; point the repo harness at its URL (the workbench
stored it in the alias). The harness measures an existing endpoint when `--endpoint` is given:

```bash
python run_matrix.py --backend vllm --endpoint https://<endpoint-url> --configurations E6
python run_robolab.py --backend vllm \
  --endpoint-baseline https://<baseline-url> --endpoint-candidate https://<optimized-url>
```

To reproduce the **full readable ablation** (R0-R4, G0-G5, E0-E6) you deploy the config you want
to compare (each endpoint bakes in one config's flags) and measure it — or run the eager PyTorch
matrix on a rented/allocated GPU (`run_matrix.py --backend vllm`, which launches its own server
per config). One serverless endpoint = one engine config, so the workbench path is best for the
**baseline-vs-final** production comparison (Job 2) and the RoboLab gate (Jobs 3-4).

## Verified facts / caveats (npa 0.1.0)

- `npa workbench cosmos optimize` and `… finetune` are **"Roadmap placeholder"** commands — we
  don't depend on them.
- `npa workbench cosmos infer` is **world-generation** (`--prompt` / `--input-path` text/image/
  video-to-world), *not* an action-chunk request. Use it only as a connectivity smoke; the real
  32×8 action request goes to the endpoint URL directly (`# VERIFY` the payload in `policy/serving.py`).
- `npa workbench cosmos train --runtime serverless` **does** create a Nebius AI Job, but runs a
  hard-coded Cosmos workload — it can't run this repo's code.
- `npa burst submit` runs an arbitrary `--entrypoint` on SkyPilot GPU nodes, but it's
  torchrun-wrapped (built for distributed training), so it's a poor fit for the ablation orchestrator.
- The `*.sky.yaml` files here are optional **raw-SkyPilot** task specs (RoboLab on RT-core GPUs,
  etc.), not npa commands — the npa path above is primary.

RoboLab (Jobs 3-4) still needs a different stack — Isaac Sim + Isaac Lab on an RT-core GPU
(`gpu-l40s-d` / `gpu-rtx6000`, not H200) — see [job3-robolab-subset.sky.yaml](job3-robolab-subset.sky.yaml).
