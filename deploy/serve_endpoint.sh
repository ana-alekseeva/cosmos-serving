#!/usr/bin/env bash
# ENTRYPOINT for cosmos-droid-vllm:v4 (SERVE variant) — run Cosmos3-Nano-Policy-DROID as a
# long-lived Nebius SERVERLESS endpoint. npa serverless launches the image ENTRYPOINT and
# injects ONLY env (COSMOS_MODEL_ID, COSMOS_SERVER_PORT=8080) — never a container command —
# so the image must start the server itself. CONFIG selects the E-ladder rung:
#   CONFIG=E0  baseline eager (TORCH_SDPA + --enforce-eager)     -> Job 3 baseline endpoint
#   CONFIG=E4  final optimized (FLASH_ATTN + compile/graphs + fp8) -> Job 3 optimized endpoint
set -euo pipefail
cd /opt/cosmos-serving

# cosmos_framework is vendored by PYTHONPATH (baked into v3 with the data/vfm + model/vfm ->
# generator symlinks the cosmos3 pipeline's lazy imports need). Keep it explicit in case a
# future base image drops the env.
export PYTHONPATH="/opt/cosmos-framework${PYTHONPATH:+:$PYTHONPATH}"
# vllm-omni pulls the gated checkpoint by id; accept either token name.
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

exec /opt/omni/.venv/bin/python deploy/serve_launch.py \
  --config "${CONFIG:-E0}" \
  --model  "${COSMOS_MODEL_ID:-nvidia/Cosmos3-Nano-Policy-DROID}" \
  --port   "${COSMOS_SERVER_PORT:-8080}"
