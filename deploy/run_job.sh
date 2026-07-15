#!/usr/bin/env bash
# Entrypoint for the Nebius benchmark jobs (`nebius ai job create`). The image (deploy/Dockerfile)
# already has BOTH venvs + cuDNN>=9.22; this stages the model + replay set, runs the requested
# matrix, aggregates, and optionally uploads to S3. Inject + run it with:
#   --inject-file deploy/run_job.sh:/run_job.sh  --container-command bash  --args /run_job.sh
#
# Env (pass via `nebius ai job create --env ... / --env-secret ...`):
#   HF_TOKEN            (secret, required)   HF login for the gated Cosmos3-Nano-Policy-DROID
#   BACKEND=pytorch     pytorch (native P ladder, deploy/Dockerfile image) | vllm (production
#                       E configs via vLLM/vLLM-Omni, deploy/Dockerfile.vllm image — Job 2:
#                       BACKEND=vllm CONFIGS=E0,E6 OUTPUT_PREFIX=production/)
#   OUTPUT_PREFIX=      appended to OUTPUT_URI (job-specific S3 subdir; .env wins over --env
#                       for OUTPUT_URI itself)
#   MODE=matrix         matrix (run_matrix.py) | multigpu (run_multigpu.py, needs >=2 GPUs) |
#                       profile (torch.profiler Chrome traces ONLY -> ${OUTPUT_URI}raw/traces/,
#                       open at https://ui.perfetto.dev; no matrix)
#   CONFIGS=            comma-sep cids, empty = full matrix   (e.g. "E0,E6")
#   PROFILE_CONFIGS=    space-sep cids to trace (profile mode default "P0 P1 P2 P3"); set it on a
#                       matrix run to ALSO upload traces after the results
#   REPLAY_N=50  REPLAY_SIZE=50  WARMUPS=5
#   OUTPUT_DIR=results
#   OUTPUT_URI=  AWS_ENDPOINT_URL=   optional S3 upload (needs AWS_* creds in env too)
#   TENSOR_PARALLEL_SIZE=  PARALLEL=  (job2b: read by policy/serving.py — # VERIFY)
set -euo pipefail
SERVING=/opt/cosmos-serving; FRAMEWORK=/opt/cosmos-framework
cd "$SERVING"
# Drive the whole job from a single injected .env (mirrors deploy/setup.sh) instead of many
# --env/--env-secret flags. .env is dockerignored (never baked into the image), so inject it at
# launch:  nebius ai job create --inject-file .env:/opt/cosmos-serving/.env ...
# Values in .env are applied to the environment (they win over any --env passed for the same key).
[ -f "$SERVING/.env" ] && { set -a; . "$SERVING/.env"; set +a; }
# .env wins over --env for OUTPUT_URI, so per-job destinations use OUTPUT_PREFIX instead
# (e.g. job 2 passes --env OUTPUT_PREFIX=production/ -> ${OUTPUT_URI}production/).
[ -n "${OUTPUT_PREFIX:-}" ] && OUTPUT_URI="${OUTPUT_URI:-}${OUTPUT_PREFIX}"
: "${HF_TOKEN:?set HF_TOKEN (inject .env, or pass nebius --env-secret HF_TOKEN=...)}"
: "${BACKEND:=pytorch}"; : "${MODE:=matrix}"; : "${CONFIGS:=}"
# This native cu130 image has NO vLLM. The E-ladder (E0-E6) routes to the vLLM backend
# (policy/compat.resolve_backend) and dies here with "No such file or directory: 'vllm'". So for the
# pytorch matrix, default to the native P ladder unless CONFIGS is set explicitly. Run the E-ladder
# on a separate --group vllm image (BACKEND=vllm CONFIGS=E0,...,E6).
if [ "$MODE" = matrix ] && [ -z "$CONFIGS" ]; then
  # vllm default = Job 2's scope (§4): production baseline + final optimized ONLY, never the
  # full matrix (P configs would route to the absent native stack on this image, and the spec
  # explicitly says not to repeat the ablation ladder in production engines).
  [ "$BACKEND" = pytorch ] && CONFIGS="P0,P1,P2,P3" || CONFIGS="E0,E6"
fi
: "${REPLAY_N:=50}"; : "${REPLAY_SIZE:=50}"; : "${WARMUPS:=5}"; : "${OUTPUT_DIR:=results}"
mkdir -p /local/replay /local/model

MODEL="$(sed -nE 's/^[[:space:]]+model:[[:space:]]*//p' config/experiment.yaml | head -1)"

# --- CUDA sanity FIRST: a broken node/cuBLAS should fail here in ~1 min, BEFORE the replay
# capture and the 30GB model download. ---
# The cu130 group pins nvidia-cublas==13.1.0.3 (torch 2.10+cu130's own pin) — BROKEN on
# Hopper/Blackwell + r580 drivers: trivial GEMMs fail with CUBLAS_STATUS_INVALID_VALUE
# (vllm#35028 is this exact stack; pytorch#174949 is the cu12 twin, fixed by a newer cuBLAS).
# 13.1.1.3 is the patched pin torch 2.13+cu130 ships — override to it. NB this is why setup.sh
# runs fine on a VM: without a cu-group it gets PyPI torch 2.10.0 = the cu128 build + cuBLAS
# 12.8.4.1, which works on the same GPU/driver.
# ORDER MATTERS: `uv sync` FIRST — it heals a stale image up to the lock (an image baked without
# --all-extras is missing mainline deps like iopath and the matrix dies on import), but it also
# PRUNES anything not in the lock. So every override (this cuBLAS pin, the cuDNN>=9.22 bump, the
# editable cosmos-serving that sync uninstalls) MUST come after it, never before.
# Names: PyPI renamed the CUDA-13 libs — nvidia-cublas-cu13 / nvidia-cuda-runtime-cu13 are dead
# 0.0.1 stubs; the real wheels are UNSUFFIXED nvidia-cublas / nvidia-cuda-runtime. cuDNN kept -cu13.
# If the pin still fails, the preflight escalates to the newest cuBLAS via LD_PRELOAD, and dumps a
# raw-cuBLAS probe (torch-side vs node-side) before giving up.
# NB: `uv pip install` targets the active $VIRTUAL_ENV — pin --python or libs land in the wrong
# venv (project-scoped `uv sync` is immune; setup.sh dodges it with `unset VIRTUAL_ENV`).
PYMODEL="$FRAMEWORK/.venv/bin/python"
if [ "$BACKEND" = vllm ]; then
  # vLLM image (deploy/Dockerfile.vllm): vllm group, NO --all-extras (extras drag in extra ABI
  # breakage). After the sync, re-apply everything it prunes/reinstalls (same rule as native):
  #   * torch -> the PyPI cu128 build. vllm 0.19.1's PyPI wheels are CUDA-12 binaries
  #     (vllm/_C.abi3.so NEEDs libcudart.so.12), but the framework maps torch to the cu13
  #     pytorch index, so the sync leaves a cu13-only venv where `import vllm._C` dies on any
  #     GPU node ("libcudart.so.12: cannot open shared object file"). PyPI torch==2.10.0 is the
  #     cu128 flavor vllm was built against, and (unlike pytorch-index wheels) it declares its
  #     nvidia-*-cu12 dep tree, which brings libcudart.so.12 + cuBLAS 12.8.4.1 (known-good on
  #     H200+r580). --reinstall-package forces the flavor swap (PEP440: ==2.10.0 is already
  #     "satisfied" by 2.10.0+cu130); --index-url keeps uv off the framework's index mapping.
  #   * vllm-omni — separate PyPI package, not in the lock; pinned 0.20.0 (0.24 force-upgrades
  #     transformers>=5.5 past what the lock resolved) — lockstep with deploy/Dockerfile.vllm.
  #   * UNINSTALL torchaudio — transformers imports it opportunistically guarded only by
  #     `except ImportError`: broken -> OSError crash on the vllm CLI path, absent -> clean skip.
  # NB: do NOT add the cu13 nvidia-cublas/nvidia-cuda-runtime pins here — those versions are
  # torch 2.13+cu130's pins and make uv upgrade torch to 2.13.0+cu130 to match (that silent
  # jump is exactly what a failed Job 2 preflight showed alongside the libcudart error).
  ( cd "$FRAMEWORK" && uv sync --group vllm )
  uv pip install -q --python "$PYMODEL" --index-url https://pypi.org/simple \
      --reinstall-package torch --reinstall-package torchvision \
      "torch==2.10.0" "torchvision==0.25.0" "vllm-omni==0.20.0"
  uv pip uninstall -q --python "$PYMODEL" torchaudio 2>/dev/null || true
  # GPU-op trace dir — OUR knob, not vLLM's: the env var was removed from vllm 0.19.1;
  # policy/serving.py translates it into the --profiler-config engine flag at server launch
  # (per-config subdirs via policy/pipeline.py). Traces upload to ${OUTPUT_URI}raw/traces/.
  export VLLM_TORCH_PROFILER_DIR="${VLLM_TORCH_PROFILER_DIR:-/local/vllm_traces}"
  mkdir -p "$VLLM_TORCH_PROFILER_DIR"
else
  ( cd "$FRAMEWORK" && uv sync --all-extras --group cu130 --group policy-server )
  uv pip install -qU --python "$PYMODEL" \
      "nvidia-cudnn-cu13>=9.22" "nvidia-cublas==13.1.1.3" "nvidia-cuda-runtime==13.0.96"
fi
uv pip install -q --python "$PYMODEL" -e "$SERVING"

# Deterministic cuBLAS (the policy server runs with deterministic_seed=True) needs a workspace
# config; without it some cuBLAS paths fail to initialize. Set globally, preflight probes WITH it
# so the probe env == the matrix env.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Put cuBLASLt's JIT kernel cache on the big /local NVMe, not the container's (possibly full)
# overlay: a full overlay makes cublasLt fail to write its cache and report NOT_INITIALIZED.
export CUDA_CACHE_PATH=/local/.nv_cache; mkdir -p "$CUDA_CACHE_PATH"

cat > /tmp/gemm_probe.py <<'PY'
import sys, torch
from importlib.metadata import version, PackageNotFoundError
def v(*names):  # native venv = cu13 wheels (unsuffixed cublas/runtime); vllm venv = -cu12 ones
    for p in names:
        try: return f"{p}=={version(p)}"
        except PackageNotFoundError: pass
    return "absent"
print("PREFLIGHT torch", torch.__version__, "toolkit_cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("PREFLIGHT wheels", v("nvidia-cublas", "nvidia-cublas-cu12"),
      v("nvidia-cudnn-cu13", "nvidia-cudnn-cu12"),
      v("nvidia-cuda-runtime", "nvidia-cuda-runtime-cu12"))
print("PREFLIGHT gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
ok = True
for dt in ("float32", "float16", "bfloat16"):
    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=getattr(torch, dt)); (a @ a).float().sum().item()
        print(f"PREFLIGHT GEMM {dt}: OK")
    except Exception as e:
        ok = False; print(f"PREFLIGHT GEMM {dt}: FAIL {type(e).__name__}: {str(e)[:110]}")
# the .so that actually got mapped (wheel vs /usr/local/cuda system copy) — metadata can't tell us
libs = sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "cublas" in ln})
print("PREFLIGHT loaded_cublas", *(libs or ["none"]))
sys.exit(0 if ok else 2)
PY

# Raw cuBLAS SGEMM via ctypes — bypasses torch's handle/workspace management entirely, so it
# separates "this cuBLAS build is broken on this node" from "torch's cuBLAS setup is broken".
# argv[1] = dir containing libcublas.so.13 ("" = let ld.so pick the system copy).
cat > /tmp/raw_cublas.py <<'PY'
import ctypes, sys
libdir = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else None
tag = libdir or "ld.so default"
cudart = ctypes.CDLL("libcudart.so.13", mode=ctypes.RTLD_GLOBAL)
if libdir:
    ctypes.CDLL(f"{libdir}/libcublasLt.so.13", mode=ctypes.RTLD_GLOBAL)
    cublas = ctypes.CDLL(f"{libdir}/libcublas.so.13", mode=ctypes.RTLD_GLOBAL)
else:
    cublas = ctypes.CDLL("libcublas.so.13", mode=ctypes.RTLD_GLOBAL)
h = ctypes.c_void_p()
st = cublas.cublasCreate_v2(ctypes.byref(h))
if st != 0:
    print(f"RAW [{tag}] cublasCreate={st} (0=OK)"); sys.exit(3)
ver = ctypes.c_int(); cublas.cublasGetVersion_v2(h, ctypes.byref(ver))
n = 512; bufs = []
for _ in range(3):
    p = ctypes.c_void_p()
    if cudart.cudaMalloc(ctypes.byref(p), n * n * 4) != 0:
        print(f"RAW [{tag}] cudaMalloc FAILED"); sys.exit(4)
    bufs.append(p)
one, zero = ctypes.c_float(1.0), ctypes.c_float(0.0)
st = cublas.cublasSgemm_v2(h, 0, 0, n, n, n, ctypes.byref(one), bufs[0], n, bufs[1], n,
                           ctypes.byref(zero), bufs[2], n)
sync = cudart.cudaDeviceSynchronize()
print(f"RAW [{tag}] cublas_version={ver.value} sgemm={st} sync={sync} (0=OK)")
sys.exit(0 if st == 0 and sync == 0 else 3)
PY

echo "== PREFLIGHT =="
df -h / /local /root/.cache 2>/dev/null | sed 's/^/PREFLIGHT df /'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/PREFLIGHT smi /'
# Image identity: a REUSED tag (e.g. :v1 pushed twice) is served stale from the node's image
# cache (IfNotPresent) — this line is how you catch it. Compare against the framework commit
# your local image reports; if they differ, the node ran an old build: push a FRESH tag.
echo "PREFLIGHT framework $(git -C "$FRAMEWORK" log -1 --format='%h %cs %s' 2>/dev/null || echo unknown)"
rc=0; "$PYMODEL" /tmp/gemm_probe.py || rc=$?
# The escalation below is cu13-only (preloads libcublas.so.13): the vllm venv is the cu128
# stack (cuBLAS 12.8.4.1, known-good) — a GEMM failure there is a node problem, go to abort.
if [ "$rc" -ne 0 ] && [ "$BACKEND" != vllm ]; then
  echo "PREFLIGHT: GEMM failed with torch's patched cuBLAS pin -> escalating to newest nvidia-cublas via LD_PRELOAD"
  uv pip install -q --python "$PYMODEL" --target /local/cublas-alt "nvidia-cublas==${CUBLAS_ALT:-13.6.0.2}"
  ALT=/local/cublas-alt/nvidia/cu13/lib
  if LD_PRELOAD="$ALT/libcublasLt.so.13:$ALT/libcublas.so.13" "$PYMODEL" /tmp/gemm_probe.py; then
    export LD_PRELOAD="$ALT/libcublasLt.so.13:$ALT/libcublas.so.13"
    echo "PREFLIGHT: GEMM OK with newest cuBLAS preloaded -> matrix will run with it"
    rc=0
  else
    echo "PREFLIGHT: GEMM fails with BOTH the pinned and the newest cuBLAS. Raw-cuBLAS isolation:"
    SP="$("$PYMODEL" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    "$PYMODEL" /tmp/raw_cublas.py "$SP/nvidia/cu13/lib" || true
    "$PYMODEL" /tmp/raw_cublas.py "$ALT" || true
    "$PYMODEL" /tmp/raw_cublas.py "" || true
  fi
fi
# Import smoke: the matrix's engine imports are lazy (policy/pytorch_engine.py) and run_matrix
# swallows per-config errors, so a missing dep (e.g. iopath) otherwise surfaces only AFTER the
# capture + 30GB download, once per config. Fail here, in seconds, with the real ImportError.
irc=0
if [ "$rc" -eq 0 ]; then
  if [ "$BACKEND" = vllm ]; then
    mods="import vllm, policy.runner, policy.serving"
    # Don't swallow the error: "CLI missing" vs "CLI crashes on import" need different fixes
    # (absent binary -> wrong image; ImportError/OSError -> broken venv, e.g. the torchaudio ABI
    # crash). Print the tail of the real traceback.
    "$FRAMEWORK/.venv/bin/vllm" --help >/dev/null 2>/tmp/vllm_cli_err || {
      echo "PREFLIGHT: vllm CLI failed — last lines:"
      tail -15 /tmp/vllm_cli_err | sed 's/^/PREFLIGHT vllm-cli /'
      irc=1
    }
  else
    mods="import iopath, policy.runner, policy.pytorch_engine, \
cosmos_framework.inference.args, cosmos_framework.scripts.action_policy_server_robolab, \
cosmos_framework.scripts.action_policy_server_utils"
  fi
  [ "$irc" -eq 0 ] && { "$PYMODEL" -c "$mods; print('PREFLIGHT imports OK')" || irc=$?; }
fi
echo "== END PREFLIGHT =="
if [ "$rc" -ne 0 ]; then
  echo "PREFLIGHT: trivial GEMM fails with every cuBLAS build. RAW lines above tell you which:"
  echo "PREFLIGHT:   raw sgemm=0 but torch FAIL -> torch-side (report with these numbers)"
  echo "PREFLIGHT:   raw sgemm!=0 too          -> node/driver problem -> retry on another node / report to Nebius"
  echo "PREFLIGHT: aborting before the 30GB model load."
  exit 1
fi
if [ "$irc" -ne 0 ]; then
  echo "PREFLIGHT: the matrix's own imports fail in the model venv (ImportError above) — the venv"
  echo "PREFLIGHT: is missing deps even after uv sync. Rebuild the image from the current Dockerfile."
  echo "PREFLIGHT: aborting before the 30GB model load."
  exit 1
fi

# --- stage the replay set in the HARNESS venv (has tfds) ---
# shellcheck disable=SC1091
source "$SERVING/.venv/bin/activate"
command -v hf >/dev/null 2>&1 || uv pip install -q huggingface_hub
hf auth login --token "$HF_TOKEN" --add-to-git-credential || true
python -m policy.capture --n "$REPLAY_N" --out /local/replay

# model weights for the native-PyTorch path (vLLM pulls by id, so only when BACKEND=pytorch)
[ "$BACKEND" = pytorch ] && hf download "$MODEL" --local-dir /local/model

# --- run the matrix in the MODEL venv (torch + cosmos_framework + policy) ---
# shellcheck disable=SC1091
source "$FRAMEWORK/.venv/bin/activate"

# MODE=profile: Perfetto/Chrome traces only — profile_and_upload.sh runs policy.profile_pytorch
# per config (1 traced request after warmups) and uploads to ${OUTPUT_URI}raw/traces/.
if [ "$MODE" = profile ]; then
  command -v aws >/dev/null 2>&1 || uv pip install -q awscli
  PROFILE_CONFIGS="${PROFILE_CONFIGS:-P0 P1 P2 P3}" bash deploy/profile_and_upload.sh /local/replay/manifest.json
  echo "DONE (profile) -> ${OUTPUT_URI:-none}raw/traces/"
  exit 0
fi

cfg=(); [ -n "$CONFIGS" ] && cfg=(--configurations "$CONFIGS")
if [ "$MODE" = multigpu ]; then
  python run_multigpu.py --backend "$BACKEND" --manifest /local/replay/manifest.json --out-dir "$OUTPUT_DIR"
else
  python run_matrix.py --backend "$BACKEND" --checkpoint-dir /local/model \
    --input-manifest /local/replay/manifest.json \
    --replay-size "$REPLAY_SIZE" --warmups "$WARMUPS" --output-dir "$OUTPUT_DIR" "${cfg[@]}"
fi
python aggregate.py --out-dir "$OUTPUT_DIR" || true

# --- optional S3 upload (Nebius object storage) ---
if [ -n "${OUTPUT_URI:-}" ]; then
  command -v aws >/dev/null 2>&1 || uv pip install -q awscli
  ep=(); [ -n "${AWS_ENDPOINT_URL:-}" ] && ep=(--endpoint-url "$AWS_ENDPOINT_URL")
  aws s3 cp "$OUTPUT_DIR/" "${OUTPUT_URI}raw/" --recursive --exclude "aggregate/*" "${ep[@]}" || true
  aws s3 cp "$OUTPUT_DIR/aggregate/" "${OUTPUT_URI}" --recursive "${ep[@]}" || true
  # vLLM server-side profiler traces, if any were flushed (needs /start_profile wiring).
  [ -n "${VLLM_TORCH_PROFILER_DIR:-}" ] && [ -n "$(ls -A "$VLLM_TORCH_PROFILER_DIR" 2>/dev/null)" ] \
    && aws s3 cp "$VLLM_TORCH_PROFILER_DIR/" "${OUTPUT_URI}raw/traces/" --recursive "${ep[@]}" || true
fi
# Optional Perfetto traces alongside a matrix run (e.g. PROFILE_CONFIGS="P0 P3").
if [ -n "${PROFILE_CONFIGS:-}" ]; then
  command -v aws >/dev/null 2>&1 || uv pip install -q awscli
  bash deploy/profile_and_upload.sh /local/replay/manifest.json || true
fi
echo "DONE -> $OUTPUT_DIR (uri: ${OUTPUT_URI:-none})"
