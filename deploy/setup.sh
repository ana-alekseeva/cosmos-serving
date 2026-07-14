#!/usr/bin/env bash
# Prepare a GPU box for a REAL run of the Cosmos3-Nano-Policy-DROID waterfall.
# Two runtimes: native PyTorch reference (§5.3.1 — the R/G configs) + vLLM/vLLM-Omni
# (§5.3.2/§5.3.3 — the end-to-end E ladder and the Cache-DiT/FP8 rungs), routed per config.
#
# Run once on the box:   bash deploy/setup.sh
# Then do the 1-sample run (commands printed at the end).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

# 1. Python env (uv manages the harness venv) + the DROID-capture dependency.
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv pip install tensorflow-datasets huggingface_hub   # DROID capture + HF CLI

# 2. Secrets from .env (gitignored). Copy .env.example -> .env and fill it in first.
if [ ! -f .env ]; then
  echo "missing .env — copy .env.example to .env and fill in HF_TOKEN (+ AWS keys for uploads)" >&2
  exit 1
fi
set -a; . ./.env; set +a                              # load HF_TOKEN etc. into this shell
: "${HF_TOKEN:?HF_TOKEN not set in .env (accept the Cosmos3-Nano-Policy-DROID license on HF first)}"
uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true

# 3. Model checkpoint for the native-PyTorch path (pytorch loads it from --checkpoint-dir; the
#    routed vLLM/vLLM-Omni configs pull it from HF by id). MODEL comes from experiment.yaml.
MODEL="$(sed -nE 's/^[[:space:]]+model:[[:space:]]*//p' config/experiment.yaml | head -1)"
uv run huggingface-cli download "$MODEL" --local-dir /local/model

# 4. vLLM + vLLM-Omni are needed for the routed E / Cache-DiT / FP8 configs (§5.3.3).
#    Recommended: the all-in-one cosmos3 image. Confirm they import (VERIFY versions):
if uv run python -c "import vllm" 2>/dev/null; then
  echo "vllm: importable"
else
  echo "WARNING: vllm not importable — install vLLM + vLLM-Omni for the E/G4/G5 configs" >&2
fi

cat <<'NEXT'

setup done. Real 1-sample run:

  # capture ONE real DROID observation (shared by every config)
  uv run python -m policy.capture --n 1 --out /local/replay

  # smoke ONE native-PyTorch config (R0) and ONE vLLM config (E0)
  uv run python run_configuration.py --configuration R0 --backend pytorch \
    --manifest /local/replay/manifest.json --checkpoint-dir /local/model \
    --replay-size 1 --warmups 0 --out-dir results-smoke
  uv run python run_configuration.py --configuration E0 --backend vllm \
    --manifest /local/replay/manifest.json --replay-size 1 --warmups 0 --out-dir results-smoke

  # then the full 1-sample matrix (auto-routes R/G->pytorch, G4/G5+E->vllm)
  uv run python run_matrix.py --backend pytorch --smoke \
    --input-manifest /local/replay/manifest.json --output-dir results-smoke
  uv run python aggregate.py --out-dir results-smoke
NEXT
