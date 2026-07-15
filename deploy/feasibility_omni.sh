#!/usr/bin/env bash
# Feasibility test: serve nvidia/Cosmos3-Nano-Policy-DROID through STOCK vLLM-Omni on an
# H200 VM. Answers, in ~6 staged checks, whether Job 2 can use the official serving path:
#
#   Cosmos3 support landed in vllm-omni 0.22 (registry: Cosmos3OmniDiffusersPipeline —
#   exactly the checkpoint's model_index.json _class_name). This test uses the CLEAN
#   0.24/0.24 pairing in a FRESH venv from PyPI: no cosmos-framework, no uv-group locks,
#   no cu12/cu13 flavor juggling — vllm's own dep tree is self-consistent.
#   The pipeline's action path: extra arg action_mode="policy" (also forward_dynamics /
#   inverse_dynamics), embodiment "droid_lerobot" (domain id 8) — RoboLab/OpenPI
#   observations bypass video output and return action-only custom output.
#
# Run ON THE VM (H200, NVIDIA driver, ~80GB free disk):
#   HF_TOKEN=hf_... bash feasibility_omni.sh 2>&1 | tee feasibility.log
#
# Stages: S1 venv+install  S2 CLI sanity  S3 serve+load  S4 /v1/models
#         S5 action request  S6 profiler roundtrip
set -uo pipefail

: "${HF_TOKEN:?export HF_TOKEN first}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
MODEL="nvidia/Cosmos3-Nano-Policy-DROID"
PORT="${PORT:-8091}"
WORK="${WORK:-$HOME/omni-feas}"
VLLM_PIN="${VLLM_PIN:-0.24.0}"        # keep vllm and vllm-omni major/minor IN LOCKSTEP
OMNI_PIN="${OMNI_PIN:-0.24.0}"
TRACE_DIR="$WORK/traces"
mkdir -p "$WORK" "$TRACE_DIR"
pass() { echo "== PASS $*"; }
fail() { echo "== FAIL $* — see $WORK/serve.log"; exit 1; }

# --- S1: fresh venv, stock PyPI stack ------------------------------------------------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv "$WORK/.venv" --python 3.12 || fail S1 venv
PY="$WORK/.venv/bin/python"
uv pip install --python "$PY" -q "vllm==$VLLM_PIN" "vllm-omni==$OMNI_PIN" || fail S1 install
"$PY" - <<'EOF' || fail S1 versions
from importlib.metadata import version
for p in ("vllm", "vllm-omni", "torch", "transformers", "diffusers"):
    print("S1", p, version(p))
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS
assert "Cosmos3OmniDiffusersPipeline" in _DIFFUSION_MODELS, "no Cosmos3 pipeline registered!"
print("S1 Cosmos3OmniDiffusersPipeline registered OK")
EOF
pass S1 "venv + Cosmos3 pipeline registered"

# --- S2: CLI sanity on a real GPU box (this is what always broke in docker build) -----
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$WORK/.venv/bin/vllm" --help >/dev/null || fail S2 "vllm CLI"
"$WORK/.venv/bin/vllm-omni" --help >/dev/null || fail S2 "vllm-omni CLI"
pass S2 "both CLIs run"

# --- S3: serve the checkpoint (downloads ~30GB on first run) --------------------------
echo "S3 launching: vllm-omni serve $MODEL (log: $WORK/serve.log)"
"$WORK/.venv/bin/vllm-omni" serve "$MODEL" \
    --port "$PORT" --max-num-seqs 1 \
    --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$TRACE_DIR\"}" \
    >"$WORK/serve.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT
t0=$SECONDS
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  kill -0 $SERVER_PID 2>/dev/null || fail S3 "server exited during startup"
  [ $((SECONDS - t0)) -gt 3600 ] && fail S3 "not healthy within 60min"
  sleep 10
done
echo "S3 healthy after $((SECONDS - t0))s; VRAM:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
pass S3 "server healthy"

# --- S4: model listed ------------------------------------------------------------------
curl -sf "http://127.0.0.1:$PORT/v1/models" | tee "$WORK/models.json" | grep -q "Cosmos3" \
  && pass S4 "/v1/models lists the model" || fail S4 "/v1/models"

# --- S5: one action request (policy mode, DROID embodiment) ---------------------------
# 64x64 red PNG as the camera frame — feasibility probes the ROUTE + action_mode plumbing,
# not output quality. If the schema is rejected, serve.log + response.json show the
# server's expected shape (the pipeline reads extra args: action_mode, embodiment, ...).
"$PY" - <<EOF | tee "$WORK/response.json"
import base64, io, json, urllib.request
try:
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (64, 64), (200, 30, 30)).save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
except ImportError:
    b64 = ""  # PIL ships with vllm; fallback keeps the probe alive
body = {
    "model": "$MODEL",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": "Pick up the banana and place it in the bowl."},
    ]}],
    "action_mode": "policy",
    "embodiment": "droid_lerobot",
}
req = urllib.request.Request("http://127.0.0.1:$PORT/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
try:
    print(urllib.request.urlopen(req, timeout=300).read().decode()[:3000])
except Exception as e:
    body_txt = getattr(e, "read", lambda: b"")()
    print("S5 request failed:", e, body_txt[:2000])
EOF
echo "S5: inspect response above — an action tensor (or a schema error naming the expected fields) both count as feasibility signal"

# --- S6: profiler roundtrip ------------------------------------------------------------
curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" && curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" \
  && ls -la "$TRACE_DIR" && pass S6 "profiler endpoints + trace dir" \
  || echo "== WARN S6 profiler endpoints not mounted (check --profiler-config propagation)"

echo "== FEASIBILITY RUN COMPLETE — artifacts: $WORK/{serve.log,models.json,response.json}, $TRACE_DIR"
