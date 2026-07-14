#!/usr/bin/env bash
# Create a Nebius SERVERLESS AI Endpoint for Cosmos3-Nano-Policy-DROID with the ablation-winning
# latency optimizations, using the REAL current npa workbench commands (verified against
# npa 0.1.0 `npa workbench cosmos deploy --help`). `--runtime serverless` "creates a Nebius
# Serverless AI Endpoint for the Cosmos container and stores its endpoint URL in the workbench
# alias" — i.e. the workbench provisions the resource; nothing is created by hand.
#
# Prereqs: bash deploy/install_npa.sh  &&  npa configure --interactive  (AI Jobs IAM role).
#
#   MODE=optimized bash jobs/deploy-optimized.sh     # E6 flags baked in (default)
#   MODE=baseline  bash jobs/deploy-optimized.sh     # eager, no opt flags (for comparison)
set -euo pipefail

MODE="${MODE:-optimized}"                          # optimized (E6) | baseline (eager)
PROJECT_ALIAS="${PROJECT_ALIAS:-cosmos}"           # -p: workbench alias in ~/.npa/config.yaml
NAME="${NAME:-cosmos-policy-$MODE}"                # -n: workbench instance name
PROJECT_ID="${PROJECT_ID:-}"                       # Nebius project id (or set in ~/.npa/config.yaml)
MODEL="$(sed -nE 's/^[[:space:]]+model:[[:space:]]*//p' config/experiment.yaml | head -1)"  # single source: experiment.yaml
GPU_TYPE="${GPU_TYPE:-gpu-h200-sxm}"               # `npa`/nebius platform id
GPU_PRESET="${GPU_PRESET:-1gpu-16vcpu-200gb}"      # 1-GPU final; 8gpu-… for CFG-Parallel scaling
SUBNET_ID="${SUBNET_ID:-}"                         # required if the project has multiple subnets
: "${HF_TOKEN:?export HF_TOKEN=... (Cosmos3-Nano-Policy-DROID license accepted on HF)}"

# The final end-to-end optimizations (E6) as non-secret container env vars (deploy has NO
# --extra-serve-args; its knobs are --backend + repeatable --env). Baseline mode omits them.
# VERIFY the exact names the cosmos serve backend reads for FP8 / Cache-DiT / conditioning cache.
ENV_ARGS=()
if [ "$MODE" = "optimized" ]; then
  ENV_ARGS=(--env POLICY_CONDITIONING_CACHE=1 --env CACHE_DIT=1 --env QUANTIZATION=fp8)
fi
[ -n "$PROJECT_ID" ] && ENV_ARGS+=(--project-id "$PROJECT_ID")
[ -n "$SUBNET_ID" ]  && ENV_ARGS+=(--subnet-id "$SUBNET_ID")

echo "==> npa workbench cosmos deploy ($MODE) -> serverless AI endpoint"
# Group opts (-p/-n) go before the subcommand. --auto-serve (default) loads the model after a
# healthy deploy, so no separate `serve` is needed. --wait blocks until RUNNING.
npa workbench cosmos -p "$PROJECT_ALIAS" -n "$NAME" deploy \
  --runtime serverless --model "$MODEL" \
  --gpu-type "$GPU_TYPE" --gpu-preset "$GPU_PRESET" \
  "${ENV_ARGS[@]}" --auth token --wait --output json

npa workbench cosmos -p "$PROJECT_ALIAS" -n "$NAME" status --output json   # expect healthy

echo "==> Smoke-test one DROID action-chunk request"
# VERIFY: the policy infer payload — 2 camera views + instruction + 8-D proprio -> 32x8 chunk.
npa workbench cosmos -p "$PROJECT_ALIAS" -n "$NAME" infer \
  --prompt "pick up the red cube and place it in the bowl" --output-format json || true

echo "==> Done. The endpoint URL is stored in the workbench alias '$NAME'."
echo "    Point the harness at it:  python run_matrix.py --backend vllm --endpoint <url> --configurations E6"
echo "    Stop billing when finished:  npa workbench cosmos -p $PROJECT_ALIAS -n $NAME teardown --yes"
