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
# Version pins come from ONE place (deploy/versions.env); env vars still override.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_here/versions.env" ] && . "$_here/versions.env"
VLLM_PIN="${VLLM_PIN:-0.24.0}"        # keep vllm and vllm-omni major/minor IN LOCKSTEP
OMNI_PIN="${OMNI_PIN:-0.24.0}"
TRACE_DIR="$WORK/traces"
mkdir -p "$WORK" "$TRACE_DIR"
pass() { echo "== PASS $*"; }
fail() { echo "== FAIL $* — see $WORK/serve.log"; exit 1; }

# --- S1: fresh venv, stock PyPI stack ------------------------------------------------
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --clear "$WORK/.venv" --python 3.12 || fail S1 venv   # idempotent reruns (uv cache keeps reinstalls fast)
PY="$WORK/.venv/bin/python"
# cosmos-guardrail: Cosmos3's guardrail init hard-requires it (NVIDIA Open Model License) —
# without it the omni orchestrator dies at startup.
uv pip install --python "$PY" -q "vllm==$VLLM_PIN" "vllm-omni==$OMNI_PIN" cosmos-guardrail || fail S1 install
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
# Triton JIT-compiles C helpers at first GPU touch; a missing gcc or Python.h kills the
# engine core AFTER the model load. Install the toolchain if absent, then probe in seconds.
PYV=$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
if ! command -v gcc >/dev/null || [ ! -f "/usr/include/python${PYV}/Python.h" ]; then
  echo "S2 installing build toolchain (gcc + python${PYV}-dev) for triton's JIT..."
  sudo apt-get update -qq && sudo apt-get install -y -qq build-essential "python${PYV}-dev" \
    || fail S2 "toolchain install — run manually: sudo apt-get install -y build-essential python${PYV}-dev"
fi
"$PY" -c "import triton; triton.runtime.driver.active.get_current_device()" \
  || fail S2 "triton JIT probe (toolchain present but compile still failing — see error above)"
# flashinfer JIT-compiles its sampling/attention kernels too — it needs ninja (pip wheel is
# fine; .venv/bin goes on PATH for the server) and nvcc. Without nvcc, disable the flashinfer
# sampler rather than dying in the post-load sampler warmup; vllm falls back to native sampling.
uv pip install --python "$PY" -q ninja || fail S2 "ninja install"
export PATH="$WORK/.venv/bin:$PATH"
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
if ! command -v nvcc >/dev/null; then
  echo "S2 WARN: no nvcc on this box — disabling the flashinfer sampler (VLLM_USE_FLASHINFER_SAMPLER=0)"
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi
pass S2 "both CLIs run + triton JIT probe + ninja"

# --- S3: serve the checkpoint (downloads ~30GB on first run) --------------------------
echo "S3 launching: vllm-omni serve $MODEL (log: $WORK/serve.log)"
# --omni is REQUIRED: without it the CLI dispatches to VANILLA vllm (AR language tower
# only — chat works, /v1/videos 404s, no diffusion/action stage at all).
"$WORK/.venv/bin/vllm-omni" serve "$MODEL" --omni \
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

# --- S5: one POLICY action request via the REAL route ---------------------------------
# POST /v1/videos/sync as multipart form (verified against the 0.24.0 source): the camera
# frame goes as the input_reference file, the Generator recipe as form fields
# (num_inference_steps/guidance_scale/flow_shift/seed — per-REQUEST, not serve flags), and
# the action plumbing via extra_params JSON merged into the pipeline's extra_args
# (action.py: action_mode + domain_name/domain_id required; DROID = droid_lerobot -> 8).
# NB /v1/chat/completions only reaches the language tower — it chats back, no actions.
# PASS = top-level "action" field shaped [32, 8]. robot_obs (proprio) intentionally absent
# here: a validation error naming its expected structure is also useful signal — Job 2's
# replay client sends the full observation.
"$PY" - <<EOF | tee "$WORK/response.json"
import io, json, mimetypes, urllib.request, uuid
from PIL import Image

buf = io.BytesIO(); Image.new("RGB", (320, 256), (200, 30, 30)).save(buf, "PNG")
boundary = uuid.uuid4().hex
fields = {
    "model": "$MODEL",
    "prompt": "Pick up the banana and place it in the bowl.",
    "num_inference_steps": "4",
    "guidance_scale": "3",
    "flow_shift": "5",
    "seed": "0",
    "num_frames": "32",   # server contract: must equal action_chunk_size (or +1)
    "extra_params": json.dumps({
        "action_mode": "policy",
        "domain_name": "droid_lerobot",
        "action_chunk_size": 32,
    }),
}
body = b""
for k, v in fields.items():
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"input_reference\"; "
         f"filename=\"frame.png\"\r\nContent-Type: image/png\r\n\r\n").encode()
body += buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
req = urllib.request.Request(
    "http://127.0.0.1:$PORT/v1/videos/sync", body, method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
    action = resp.get("action") or resp.get("actions")
    if action is not None:
        import numpy as np
        shape = np.asarray(action).shape
        print(f"S5 action shape: {shape}")
        assert tuple(shape)[-2:] == (32, 8), f"expected [...,32,8], got {shape}"
        print("S5 PASS: action chunk [32, 8] returned")
    else:
        print("S5 no action field; top-level keys:", sorted(resp))
        print(json.dumps(resp)[:2000])
except Exception as e:
    print("S5 request failed:", e, getattr(e, "read", lambda: b"")()[:2000])
EOF
echo "S5: an action [32,8] = full pass; a validation error naming robot_obs/observation fields = the contract for the Job 2 replay client"

# --- S6: profiler roundtrip ------------------------------------------------------------
curl -sf -X POST "http://127.0.0.1:$PORT/start_profile" && curl -sf -X POST "http://127.0.0.1:$PORT/stop_profile" \
  && ls -la "$TRACE_DIR" && pass S6 "profiler endpoints + trace dir" \
  || echo "== WARN S6 profiler endpoints not mounted (check --profiler-config propagation)"

echo "== FEASIBILITY RUN COMPLETE — artifacts: $WORK/{serve.log,models.json,response.json}, $TRACE_DIR"
