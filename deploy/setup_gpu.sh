#!/usr/bin/env bash
# One-time setup on a rented GPU host (H200: 1x reasoner / 2x generator) for the
# REAL backend (--backend vllm). Written against documented interfaces; verify on-box.
set -euo pipefail

# 1. System / Python (uv manages the harness venv).
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 2. Real-backend deps (NOT installable on macOS; H200 only).
#    Package names/versions to VERIFY against current releases:
uv pip install torch vllm                 # Reasoner serving
uv pip install vllm-omni                  # Generator serving (Cosmos3 support >= 0.22.0)
uv pip install aiperf                      # NVIDIA serving benchmark client

# 3. Weights access (Cosmos3-Nano is gated — accept the license first):
#    - HF: huggingface-cli login   (then accept terms on the model page)
#    - NGC: set NGC_API_KEY if pulling from NGC
: "${HF_TOKEN:?export HF_TOKEN=... after accepting the Cosmos3-Nano license on HF}"

# 4. Smoke-test the server manually before the ablation (Reasoner, 1 GPU):
#    vllm serve nvidia/Cosmos3-Nano --host 0.0.0.0 --port 8000 --init-timeout 1800
#    curl http://127.0.0.1:8000/health
#    Generator (2 GPUs): add --omni --tensor-parallel-size 2

echo
echo "Setup done. Run the REAL ablation (harness manages the server per variant):"
echo "  uv run python -m optimize.cli --tower reasoner  --ablate --backend vllm --out-dir results"
echo "  uv run python -m optimize.cli --tower generator --ablate --backend vllm --out-dir results"
echo
echo "First verify flag mappings marked '# VERIFY' in bench/serving.py and bench/aiperf.py"
echo "against your installed vLLM / vLLM-Omni / AIPerf versions."
