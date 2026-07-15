#!/usr/bin/env bash
# Rung-by-rung feasibility on the warm VM — closes the two Job 2 caveats without cloud jobs:
#   1. robot_obs schema: every request sends the VERIFIED observation keys
#      (prompt + observation/joint_position [T,7] + observation/gripper_position [T,1];
#      camera via input_reference fallback) — exercising the RoboLab policy path for real.
#   2. engine flags: launches one server PER E-rung with the flags policy/serving.py's
#      engine_args() actually generates (harness installed into the venv — no drift), and
#      runs one action request against each.
#
# Run ON THE VM after deploy/feasibility_omni.sh has passed (reuses its venv + HF cache):
#   HF_TOKEN=hf_... bash deploy/feasibility_rungs.sh 2>&1 | tee rungs.log
#   RUNGS="E0 E1" ... to test a subset. NB: E2/E3 include torch.compile — their startup
#   can take many extra minutes; that startup cost is itself a finding.
set -uo pipefail

: "${HF_TOKEN:?export HF_TOKEN first}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
MODEL="nvidia/Cosmos3-Nano-Policy-DROID"
PORT="${PORT:-8093}"
WORK="${WORK:-$HOME/omni-feas}"
REPO="${REPO:-$HOME/cosmos-serving}"
RUNGS="${RUNGS:-E0 E1 E2 E3 E4 E5 E6}"
PY="$WORK/.venv/bin/python"
export PATH="$WORK/.venv/bin:$PATH"
mkdir -p "$WORK/rungs"

[ -x "$PY" ] || { echo "run deploy/feasibility_omni.sh first (venv missing)"; exit 1; }
uv pip install -q --python "$PY" -e "$REPO" iopath || { echo "harness install failed"; exit 1; }

# robot_obs activates the RoboLab path -> lazy cosmos_framework imports. Vendor the clone by
# PYTHONPATH (mirrors Dockerfile.vllm: vfm->generator symlink shim; iopath installed above).
FW="$WORK/cosmos-framework"
[ -d "$FW" ] || git clone --depth 1 https://github.com/NVIDIA/cosmos-framework "$FW"
[ -e "$FW/cosmos_framework/data/vfm" ] || ln -s generator "$FW/cosmos_framework/data/vfm"
[ -e "$FW/cosmos_framework/model/vfm" ] || ln -s generator "$FW/cosmos_framework/model/vfm"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$FW"
"$PY" -c "import cosmos_framework.data.vfm.action.transforms" \
  || { echo "vendored cosmos_framework import failed"; exit 1; }

# one action request with the VERIFIED robot_obs schema; prints total/stage timing
cat > "$WORK/rungs/probe.py" <<'PROBE'
import io, json, sys, time, urllib.request, uuid
from PIL import Image
import numpy as np

port, rung = sys.argv[1], sys.argv[2]
wrist = np.full((256, 320, 3), 60, dtype=np.uint8)
ext = np.full((128, 160, 3), (200, 30, 30), dtype=np.uint8)
frame = np.concatenate([wrist, np.concatenate([ext, ext], axis=1)], axis=0)
buf = io.BytesIO(); Image.fromarray(frame).save(buf, "PNG")
boundary = uuid.uuid4().hex
fields = {
    "model": "nvidia/Cosmos3-Nano-Policy-DROID",
    "prompt": "Pick up the banana and place it in the bowl.",
    "num_inference_steps": "4", "guidance_scale": "3", "flow_shift": "5",
    "seed": "0", "num_frames": "32",
    "extra_params": json.dumps({
        "action_mode": "policy", "domain_name": "droid_lerobot",
        "action_chunk_size": 32, "raw_action_dim": 8,
        "robot_obs": {
            "prompt": "Pick up the banana and place it in the bowl.",
            "observation/joint_position": [[0.1, -0.2, 0.3, -1.5, 0.0, 1.2, 0.4]],
            "observation/gripper_position": [[0.8]],
        },
    }),
}
body = b""
for k, v in fields.items():
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"input_reference\"; "
         f"filename=\"frame.png\"\r\nContent-Type: image/png\r\n\r\n").encode()
body += buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
t0 = time.perf_counter()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/videos", body, method="POST",
                             headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
ref = json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
deadline = time.time() + 900
while time.time() < deadline:
    job = json.loads(urllib.request.urlopen(
        f"http://127.0.0.1:{port}/v1/videos/{ref['id']}", timeout=30).read().decode())
    s = str(job.get("status", "")).lower()
    if "completed" in s or "failed" in s:
        break
    time.sleep(0.25)
wall = time.perf_counter() - t0
if "completed" not in s:
    print(f"RUNG {rung} REQUEST FAILED: {job.get('error')}"); sys.exit(1)
a = job.get("action") or {}
shape = tuple((a.get("shape") or [])) if isinstance(a, dict) else np.asarray(a).shape
assert tuple(shape)[-2:] == (32, 8), f"bad action shape {shape}"
print(f"RUNG {rung} OK  wall={wall:.2f}s  server={job.get('inference_time_s'):.2f}s  "
      f"stages={json.dumps(job.get('stage_durations'))}  action={shape}")
PROBE

for rung in $RUNGS; do
  flags=$("$PY" - "$rung" <<'FLAGS'
import sys, warnings
warnings.filterwarnings("ignore")
from policy.configs import config_by_id
from policy.serving import engine_args
print(" ".join(engine_args(config_by_id(sys.argv[1]))))
FLAGS
  ) || { echo "== FAIL $rung: engine_args() errored"; continue; }
  echo "== RUNG $rung flags: $flags"
  log="$WORK/rungs/serve_$rung.log"
  # shellcheck disable=SC2086
  "$WORK/.venv/bin/vllm-omni" serve "$MODEL" --omni $flags --port "$PORT" >"$log" 2>&1 &
  pid=$!
  t0=$SECONDS; up=""
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "== FAIL $rung: server died at startup (likely a flag) — tail $log:"; tail -8 "$log"; break; }
    [ $((SECONDS - t0)) -gt 2400 ] && { echo "== FAIL $rung: not healthy in 40min"; break; }
    sleep 5
  done
  if kill -0 $pid 2>/dev/null && curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "   startup: $((SECONDS - t0))s, VRAM: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    "$PY" "$WORK/rungs/probe.py" "$PORT" "$rung" || { echo "== FAIL $rung request — tail $log:"; tail -8 "$log"; }
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  sleep 3
done
echo "== RUNG SWEEP COMPLETE — logs in $WORK/rungs/"
