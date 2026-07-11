#!/usr/bin/env bash
# One-time setup on a rented GPU host (H200: 1x reasoner / 2x generator) for the
# REAL backend (--backend vllm). Written against documented interfaces; verify on-box.
set -euo pipefail

# 1. System / Python (uv manages the harness venv).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"   # uv installs here; put it on PATH for this session
uv sync

# 2. Real-backend deps (NOT installable on macOS; GPU host only).
#    IMPORTANT: vLLM and vLLM-Omni MUST be the same major.minor version, and
#    Cosmos3 needs vllm-omni >= 0.22.0. Assembling matching versions by hand is
#    fragile — the RECOMMENDED path is NVIDIA/vLLM's all-in-one Cosmos3 image:
#
#        docker run --gpus all -it -v "$PWD":/work vllm/vllm-omni:cosmos3
#        # then inside the container: install uv + `uv sync` + `uv pip install aiperf`,
#        # and run the ablation (vllm + vllm-omni + cosmos3 are already present).
#
#    Bare-metal pip fallback (pin matching versions — VERIFY the exact ones):
uv pip install torch
uv pip install "vllm==0.24.0"                          # VERIFY the version paired with cosmos3
uv pip install "vllm-omni==0.24.0"                     # MUST match vllm major.minor
uv pip install aiperf                                   # NVIDIA/ai-dynamo benchmark client (PyPI)

# 3. Weights access (Cosmos3-Nano is gated — accept the license first):
#    - HF: huggingface-cli login   (then accept terms on the model page)
#    - NGC: set NGC_API_KEY if pulling from NGC
: "${HF_TOKEN:?export HF_TOKEN=... after accepting the Cosmos3-Nano license on HF}"

# 4. Smoke-test the server manually before the ablation (Reasoner, 1 GPU):
#    vllm serve nvidia/Cosmos3-Nano --host 0.0.0.0 --port 8000
#    curl http://127.0.0.1:8000/health
#    Generator (2 GPUs): add --omni --tensor-parallel-size 2

echo
echo "Setup done. Run the REAL ablation (harness manages the server per variant):"
echo "  uv run python -m optimize.cli --tower reasoner  --ablate --backend vllm --out-dir results"
echo "  uv run python -m optimize.cli --tower generator --ablate --backend vllm --out-dir results"
echo
echo "First verify flag mappings marked '# VERIFY' in bench/serving.py and bench/aiperf.py"
echo "against your installed vLLM / vLLM-Omni / AIPerf versions."
