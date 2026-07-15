#!/usr/bin/env bash
# Create a Nebius SERVERLESS AI Endpoint for Cosmos3-Nano-Policy-DROID with the ablation-winning
# latency optimizations, using the REAL current npa workbench commands (verified against
# npa 0.1.0 `npa workbench cosmos deploy --help`). `--runtime serverless` "creates a Nebius
# Serverless AI Endpoint for the Cosmos container and stores its endpoint URL in the workbench
# alias" — i.e. the workbench provisions the resource; nothing is created by hand.
#
# Prereqs: bash deploy/install_npa.sh  &&  npa configure --interactive  (AI Jobs IAM role).
#
#   MODE=optimized bash jobs/deploy-optimized.sh     # CONFIG=E4 (FP8 final)   -> Job 3 candidate
#   MODE=baseline  bash jobs/deploy-optimized.sh     # CONFIG=E0 (eager)       -> Job 3 baseline
#
# Serves the cosmos-droid-vllm:v5 SERVE image (deploy/Dockerfile.serve): its ENTRYPOINT runs
# `vllm-omni serve --omni <engine_args(CONFIG)> --stage-overrides <policy_server_config>`, which
# exposes BOTH /v1/videos (latency) and ws /v1/realtime/robot/openpi (RoboLab) from one process.
set -euo pipefail

MODE="${MODE:-optimized}"                          # optimized (E4) | baseline (E0 eager)
PROJECT_ALIAS="${PROJECT_ALIAS:-eu-north1}"        # -p: workbench alias in ~/.npa/config.yaml (carries project_id + cosmos-droid registry)
NAME="${NAME:-cosmos-policy-$MODE}"                # -n: workbench instance name
PROJECT_ID="${PROJECT_ID:-}"                       # Nebius project id (or set in ~/.npa/config.yaml)
MODEL="$(sed -nE 's/^[[:space:]]+model:[[:space:]]*//p' config/experiment.yaml | head -1)"  # single source: experiment.yaml
GPU_TYPE="${GPU_TYPE:-gpu-h100-sxm}"               # `npa`/nebius platform id (npa GPU_TYPE_DEFAULTS: gpu-h100-sxm | gpu-h200-sxm | gpu-l40s-d | gpu-rtx6000)
GPU_PRESET="${GPU_PRESET:-1gpu-16vcpu-200gb}"      # one-H100 preset; 8gpu-… only for separate scaling experiments
SUBNET_ID="${SUBNET_ID:-}"                         # required if the project has multiple subnets
# AUTH=none for eval endpoints: neither the harness HTTP client (policy/serving.py) nor
# RoboLab's OpenPI websocket client sends an Authorization header, so `token` 401s every
# eval request. Default stays token for anything longer-lived; tear eval endpoints down.
AUTH="${AUTH:-none}"                               # eval endpoints: none (RoboLab/harness send no auth header)
# The SERVE image built from deploy/Dockerfile.serve, pushed to OUR registry (cosmos-droid,
# e00k6drmprp0pm6zcf) with a FRESH tag (never reuse tags — the cluster serves stale layers on a
# reused tag). It is cosmos-droid-vllm:v3 (the Job-2 vLLM-Omni stack) + a policy-server ENTRYPOINT.
# NOT npa-cosmos (that image is npa's stock Text2World video server — no vllm_omni, wrong API).
IMAGE="${IMAGE:-cr.eu-north1.nebius.cloud/e00k6drmprp0pm6zcf/cosmos-droid-vllm:v5}"
: "${HF_TOKEN:?export HF_TOKEN=... (Cosmos3-Nano-Policy-DROID license accepted on HF)}"

# The image ENTRYPOINT reads CONFIG and builds the full serve command via policy.serving —
# npa deploy has no --extra-serve-args, so ALL engine flags come from the image, selected by
# this one env var. E0 = TORCH_SDPA + --enforce-eager; E4 = FLASH_ATTN + compile/CUDA graphs + fp8.
CONFIG=$([ "$MODE" = "optimized" ] && echo E4 || echo E0)
ENV_ARGS=(--env "CONFIG=$CONFIG")
[ -n "$PROJECT_ID" ] && ENV_ARGS+=(--project-id "$PROJECT_ID")
[ -n "$SUBNET_ID" ]  && ENV_ARGS+=(--subnet-id "$SUBNET_ID")
[ -n "$IMAGE" ]      && ENV_ARGS+=(--image "$IMAGE")
# REPLACE=1 to delete + recreate an existing serverless alias (npa refuses to overwrite one
# otherwise). Needed to redeploy the SAME alias — e.g. after a failed deploy left it registered,
# or to roll the image tag. Note: this tears down the running endpoint before recreating it.
[ -n "${REPLACE:-}" ] && ENV_ARGS+=(--replace)

echo "==> npa workbench cosmos deploy ($MODE) -> serverless AI endpoint"
# Group opts (-p/-n) go before the subcommand. --auto-serve (default) loads the model after a
# healthy deploy, so no separate `serve` is needed. --wait blocks until RUNNING.
npa workbench cosmos -p "$PROJECT_ALIAS" -n "$NAME" deploy \
  --runtime serverless --model "$MODEL" \
  --gpu-type "$GPU_TYPE" --gpu-preset "$GPU_PRESET" \
  "${ENV_ARGS[@]+"${ENV_ARGS[@]}"}" --auth "$AUTH" --wait --output json

npa workbench cosmos -p "$PROJECT_ALIAS" -n "$NAME" status --output json   # expect healthy

echo "==> Smoke-test one DROID action-chunk request"
# VERIFY: the policy infer payload — 2 camera views + instruction + 8-D proprio -> 32x8 chunk.
npa workbench cosmos -p "$PROJECT_ALIAS" -n "$NAME" infer \
  --prompt "pick up the red cube and place it in the bowl" --output-format json || true

echo "==> Done. The endpoint URL is stored in the workbench alias '$NAME'."
echo "    Point the harness at it:  python run_matrix.py --backend vllm --endpoint <url> --configurations $CONFIG"
echo "    Stop billing when finished:  npa workbench cosmos -p $PROJECT_ALIAS -n $NAME teardown --yes"
