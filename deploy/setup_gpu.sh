#!/usr/bin/env bash
# One-time setup on a rented GPU host for the REAL backend (--backend vllm) of the
# Cosmos3-Nano-Policy-DROID ablation. Reasoner via vLLM (Qwen3-VL path), Generator / full
# policy via vLLM-Omni. Written against documented interfaces; verify on-box.
set -euo pipefail

# 1. System / Python (uv manages the harness venv).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync

# 2. Real-backend deps (NOT installable on macOS; GPU host only).
#    vLLM and vLLM-Omni MUST be the same major.minor version. The RECOMMENDED path is the
#    all-in-one Cosmos3 image (torch + vllm + vllm-omni + cosmos3 preinstalled):
#
#        docker run --gpus all -it -v "$PWD":/work vllm/vllm-omni:cosmos3
#        # inside: install uv + `uv sync`, then run the matrix.
#
#    Bare-metal pip fallback (pin matching versions — VERIFY the exact ones):
uv pip install torch
uv pip install "vllm==0.24.0"          # VERIFY the version paired with cosmos3
uv pip install "vllm-omni==0.24.0"     # MUST match vllm major.minor
uv pip install cosmos-guardrail        # REQUIRED Generator safety checker (license guard)

# 3. Weights access (Cosmos3-Nano-Policy-DROID is gated — accept the license first):
: "${HF_TOKEN:?export HF_TOKEN=... after accepting the Cosmos3-Nano-Policy-DROID license on HF}"
huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential || true

# 4. Stage the fixed replay set locally BEFORE timing (§8: "Stage all inputs ... before timing").
#    This stages the committed synthetic fixture for a MOCK dry-run. For a faithful REAL run,
#    capture 50 real DROID observations instead (same set feeds every config, §5):
#      uv pip install tensorflow-datasets
#      uv run python -m policy.capture --n 50 --out /local/replay
mkdir -p /local/replay
cp policy/mock/manifest.json /local/replay/manifest.json

# 5. Smoke-test one configuration, then run the matrix.
echo
echo "Setup done. Dry-run the plumbing with the mock backend:"
echo "  uv run python run_configuration.py --configuration E6 --backend mock --out-dir /tmp/smoke"
echo
echo "Run the REAL single-GPU ablation matrix (§4 Job 1), then aggregate (Job 5):"
echo "  uv run python run_matrix.py --config config/experiment.yaml \\"
echo "      --input-manifest /local/replay/manifest.json --output-dir results --backend vllm"
echo "  uv run python aggregate.py --out-dir results"
echo
echo "First confirm every '# VERIFY' in policy/serving.py (engine flags, static shapes for"
echo "CUDA-graph configs, the per-stage timing response) against your vLLM / vLLM-Omni versions."
