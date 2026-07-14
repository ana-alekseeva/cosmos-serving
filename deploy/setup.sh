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

# 0. Fast local scratch: model weights + the replay set stage under /local — the path every
#    job YAML and config default expects. On SkyPilot boxes /local is a pre-mounted NVMe; on a
#    hand-provisioned box it isn't, so create it once and hand it to the current user.
if [ ! -w /local ]; then
  sudo mkdir -p /local && sudo chown "$(id -un):$(id -gn)" /local
fi

# 1. Install uv (Python + venv manager) if it isn't already present.
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2. Create the harness venv and ACTIVATE it (so the rest of this script uses it directly).
uv sync                                              # creates .venv/ with the harness deps
# shellcheck disable=SC1091
set +u; source .venv/bin/activate; set -u            # activate .venv for this script
uv pip install tensorflow-datasets huggingface_hub   # DROID capture + HF CLI

# 3. Secrets from .env (gitignored). Copy .env.example -> .env and fill it in first.
if [ ! -f .env ]; then
  echo "missing .env — copy .env.example to .env and fill in HF_TOKEN (+ AWS keys for uploads)" >&2
  exit 1
fi
set -a; . ./.env; set +a                             # load HF_TOKEN etc. into this shell
: "${HF_TOKEN:?HF_TOKEN not set in .env (accept the Cosmos3-Nano-Policy-DROID license on HF first)}"
hf auth login --token "$HF_TOKEN" --add-to-git-credential || true

# 4. Model checkpoint for the native-PyTorch path (pytorch loads it from --checkpoint-dir; the
#    routed vLLM/vLLM-Omni configs pull it from HF by id). MODEL comes from experiment.yaml.
MODEL="$(sed -nE 's/^[[:space:]]+model:[[:space:]]*//p' config/experiment.yaml | head -1)"
hf download "$MODEL" --local-dir /local/model

# 5. vLLM + vLLM-Omni are needed for the routed E / Cache-DiT / FP8 configs (§5.3.3).
#    Recommended: the all-in-one cosmos3 image. Confirm they import (VERIFY versions):
if python -c "import vllm" 2>/dev/null; then
  echo "vllm: importable"
else
  echo "WARNING: vllm not importable — install vLLM + vLLM-Omni for the E/G4/G5 configs" >&2
fi

cat <<'NEXT'

setup done. Activate the venv in THIS shell, then do the 1-sample run:

  source .venv/bin/activate

  # capture ONE real DROID observation (shared by every config)
  python -m policy.capture --n 1 --out /local/replay

  # smoke ONE native-PyTorch config (R0) and ONE vLLM config (E0)
  python run_configuration.py --configuration R0 --backend pytorch \
    --manifest /local/replay/manifest.json --checkpoint-dir /local/model \
    --replay-size 1 --warmups 0 --out-dir results-smoke
  python run_configuration.py --configuration E0 --backend vllm \
    --manifest /local/replay/manifest.json --replay-size 1 --warmups 0 --out-dir results-smoke

  # then the full 1-sample matrix (auto-routes R/G->pytorch, G4/G5+E->vllm)
  python run_matrix.py --backend pytorch --smoke \
    --input-manifest /local/replay/manifest.json --output-dir results-smoke
  python aggregate.py --out-dir results-smoke
NEXT
