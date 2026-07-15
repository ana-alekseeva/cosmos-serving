#!/usr/bin/env bash
# Serve E0 (production BASELINE — eager, no optimizations) on a bare H100/H200 VM through the
# STOCK vLLM-Omni stack, exposing BOTH eval routes from one process:
#   * POST /v1/videos            — the latency harness (run_matrix.py --endpoint)
#   * ws   /v1/realtime/robot/openpi — RoboLab rollouts (Job 3)
#
# This is Path B: no npa, no serverless image, no registry. You are the data plane.
# Verified stack: vllm 0.24.0 + vllm-omni 0.24.0 (deploy/versions.env, H100 feasibility run).
#
# Prereqs on the VM: NVIDIA driver, ~80 GB free disk, this repo cloned, HF_TOKEN exported
# (Cosmos3-Nano-Policy-DROID license accepted on HF), a PUBLIC ip if the RoboLab Isaac box
# must reach this endpoint over the network.
#
#   HF_TOKEN=hf_... bash deploy/serve_e0_vm.sh          # foreground; Ctrl-C to stop
#
# Then, from anywhere that can reach the VM:
#   export E0_URL=http://<vm-ip>:8000
#   bash deploy/serve_e0_vm.sh --probe                  # health + both routes (separate shell)
set -uo pipefail

: "${HF_TOKEN:?export HF_TOKEN first (HF license accepted)}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

MODEL="nvidia/Cosmos3-Nano-Policy-DROID"
PORT="${PORT:-8000}"
WORK="${WORK:-$HOME/omni-e0}"
VLLM_PIN="${VLLM_PIN:-0.24.0}"          # keep vllm and vllm-omni major/minor IN LOCKSTEP
OMNI_PIN="${OMNI_PIN:-0.24.0}"
FW_REPO="${COSMOS_FRAMEWORK_REPO:-https://github.com/NVIDIA/cosmos-framework}"
FW_REF="${COSMOS_FRAMEWORK_REF:-main}"
FW_DIR="$WORK/cosmos-framework"
PY="$WORK/.venv/bin/python"

# ---- --probe: check the running endpoint from another shell -----------------------------
if [ "${1:-}" = "--probe" ]; then
  U="${E0_URL:?export E0_URL=http://<vm-ip>:$PORT}"
  echo "== /health";     curl -sf "$U/health" && echo " OK" || echo " FAIL"
  echo "== /v1/models";  curl -sf "$U/v1/models" | head -c 300; echo
  echo "== OpenPI ws (expect HTTP/1.1 101):"
  curl -si -H "Connection: Upgrade" -H "Upgrade: websocket" -H "Sec-WebSocket-Version: 13" \
    -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" "$U/v1/realtime/robot/openpi" | head -1
  echo "   (404 -> stage-overrides/cosmos_framework not active; the RoboLab route is missing)"
  exit 0
fi

mkdir -p "$WORK"

# --- 1. fresh venv, stock PyPI stack (feasibility S1) ------------------------------------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear "$WORK/.venv" --python 3.12
# cosmos-guardrail: Cosmos3's guardrail init hard-requires it or the orchestrator dies at start.
uv pip install --python "$PY" -q "vllm==$VLLM_PIN" "vllm-omni==$OMNI_PIN" cosmos-guardrail
# torchaudio must be UNINSTALLED: transformers>=5 (pulled by vllm-omni) imports it eagerly and
# its OSError escapes the ImportError guard, killing the omni engine after model load.
uv pip uninstall --python "$PY" torchaudio >/dev/null 2>&1 || true
"$PY" - <<'EOF'
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
assert "Cosmos3OmniDiffusersPipeline" in _DIFFUSION_MODELS, "no Cosmos3 pipeline registered!"
print("stack OK — Cosmos3OmniDiffusersPipeline registered")
EOF

# --- 2. build toolchain for triton/flashinfer JIT (feasibility S2) -----------------------
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if ! command -v gcc >/dev/null || [ ! -f "/usr/include/python${PYV}/Python.h" ]; then
  sudo apt-get update -qq && sudo apt-get install -y -qq build-essential "python${PYV}-dev"
fi
uv pip install --python "$PY" -q ninja
export PATH="$WORK/.venv/bin:$PATH"
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
command -v nvcc >/dev/null || export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc -> native sampler

# --- 3. vendor cosmos_framework by PYTHONPATH (the OpenPI/robot_obs action path needs it) -
# vllm-omni 0.24's lazy imports use the OLD layout (…data.vfm.…, …model.vfm.…); the repo
# renamed vfm -> generator, so symlink the old names onto the new dirs (no-op if unchanged).
if [ ! -d "$FW_DIR" ]; then
  git clone --depth 1 --branch "$FW_REF" "$FW_REPO" "$FW_DIR"
  [ -e "$FW_DIR/cosmos_framework/data/vfm" ]  || ln -s generator "$FW_DIR/cosmos_framework/data/vfm"
  [ -e "$FW_DIR/cosmos_framework/model/vfm" ] || ln -s generator "$FW_DIR/cosmos_framework/model/vfm"
fi
export PYTHONPATH="$FW_DIR${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline; \
  print('cosmos_framework import OK (RoboLab action transforms resolve)')" \
  || { echo "FAIL: cosmos_framework not importable — RoboLab route will 500"; exit 1; }

# --- 4. OpenPI policy-server stage override (mounts /v1/realtime/robot/openpi) ------------
# DROID: wrist + 2 external cameras, joint-position action space (matches E0/joint_pos).
cat > "$WORK/stage_overrides.json" <<'JSON'
{
  "0": {
    "model_config": {
      "policy_server_config": {
        "image_resolution": [540, 640],
        "n_external_cameras": 2,
        "needs_wrist_camera": true,
        "needs_stereo_camera": false,
        "needs_session_id": true,
        "action_space": "joint_position"
      }
    }
  }
}
JSON

# --- 5. serve with E0's exact engine flags (policy/serving.py::engine_args, all techniques
# OFF) + --omni (selects the multi-stage omni server; without it vllm serves the AR tower
# only and /v1/videos 404s). Downloads ~30 GB on first run. ------------------------------
echo "== serving E0 baseline: vllm-omni serve $MODEL --omni (eager) on :$PORT"
echo "== log: $WORK/serve.log  |  framework: $FW_DIR ($FW_REF)"
exec vllm-omni serve "$MODEL" --omni \
  --max-num-seqs 1 \
  --diffusion-attention-config.per_role.self.backend TORCH_SDPA \
  --enforce-eager \
  --host 0.0.0.0 --port "$PORT" \
  --stage-overrides "$(cat "$WORK/stage_overrides.json")" \
  2>&1 | tee "$WORK/serve.log"
