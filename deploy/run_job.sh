#!/usr/bin/env bash
# Entrypoint for the Nebius benchmark jobs (`nebius ai job create`). The image (deploy/Dockerfile)
# already has BOTH venvs + cuDNN>=9.22; this stages the model + replay set, runs the requested
# matrix, aggregates, and optionally uploads to S3. Inject + run it with:
#   --inject-file deploy/run_job.sh:/run_job.sh  --container-command bash  --args /run_job.sh
#
# Env (pass via `nebius ai job create --env ... / --env-secret ...`):
#   HF_TOKEN            (secret, required)   HF login for the gated Cosmos3-Nano-Policy-DROID
#   BACKEND=pytorch     pytorch (P->pytorch, E->vLLM) | vllm (everything on vLLM/vLLM-Omni)
#   MODE=matrix         matrix (run_matrix.py) | multigpu (run_multigpu.py, needs >=2 GPUs)
#   CONFIGS=            comma-sep cids, empty = full matrix   (e.g. "E0,E6")
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
: "${HF_TOKEN:?set HF_TOKEN (inject .env, or pass nebius --env-secret HF_TOKEN=...)}"
: "${BACKEND:=pytorch}"; : "${MODE:=matrix}"; : "${CONFIGS:=}"
# This native cu130 image has NO vLLM. The E-ladder (E0-E6) routes to the vLLM backend
# (policy/compat.resolve_backend) and dies here with "No such file or directory: 'vllm'". So for the
# pytorch matrix, default to the native P ladder unless CONFIGS is set explicitly. Run the E-ladder
# on a separate --group vllm image (BACKEND=vllm CONFIGS=E0,...,E6).
if [ "$MODE" = matrix ] && [ "$BACKEND" = pytorch ] && [ -z "$CONFIGS" ]; then
  CONFIGS="P0,P1,P2,P3"
fi
: "${REPLAY_N:=50}"; : "${REPLAY_SIZE:=50}"; : "${WARMUPS:=5}"; : "${OUTPUT_DIR:=results}"
mkdir -p /local/replay /local/model

MODEL="$(sed -nE 's/^[[:space:]]+model:[[:space:]]*//p' config/experiment.yaml | head -1)"

# --- stage the replay set in the HARNESS venv (has tfds) ---
# shellcheck disable=SC1091
source "$SERVING/.venv/bin/activate"
command -v hf >/dev/null 2>&1 || uv pip install -q huggingface_hub
hf auth login --token "$HF_TOKEN" --add-to-git-credential || true
python -m policy.capture --n "$REPLAY_N" --out /local/replay

# model weights for the native-PyTorch path (vLLM pulls by id, so only when BACKEND=pytorch)
[ "$BACKEND" = pytorch ] && hf download "$MODEL" --local-dir /local/model

# --- run in the MODEL venv (torch + cosmos_framework + policy) ---
# Self-heal images baked with the cu130 uv group: it re-pins the nvidia-cu13 CUDA libs
# (cuDNN 9.15.1 + the group's cuBLAS) to a set that fails cuBLASLt init on H200
# (CUBLAS_STATUS_NOT_INITIALIZED at the first GEMM). Re-sync WITHOUT the cu130 group (matches
# deploy/setup.sh, the build that produced the working results-real) so the current registry image
# runs with no rebuild. Idempotent: a near no-op once the image is rebuilt from the fixed Dockerfile.
( cd "$FRAMEWORK" && uv sync --all-extras --group policy-server \
    && uv pip install -qU "nvidia-cudnn-cu13>=9.22" )
# shellcheck disable=SC1091
source "$FRAMEWORK/.venv/bin/activate"

# Deterministic cuBLAS (the policy server runs with deterministic_seed=True) needs a workspace
# config; without it some cuBLAS paths fail to initialize. Cheap + safe — a real candidate fix for
# the CUBLAS_STATUS_NOT_INITIALIZED at the first GEMM. Set globally for the matrix.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Put cuBLASLt's JIT kernel cache on the big /local NVMe, not the container's (possibly full)
# overlay: a full overlay makes cublasLt fail to write its cache and report NOT_INITIALIZED.
export CUDA_CACHE_PATH=/local/.nv_cache; mkdir -p "$CUDA_CACHE_PATH"

# Preflight: disk/mem are fine but a TRIVIAL matmul fails with a cuBLAS error, so this is a
# CUDA-13/driver/cuBLAS-library mismatch on the node, not the harness. Capture the driver + nvidia-*
# wheel versions and a per-dtype GEMM result so we know WHICH mismatch (too-old driver vs bad wheel).
echo "== PREFLIGHT =="
df -h / /local /root/.cache 2>/dev/null | sed 's/^/PREFLIGHT df /'
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/PREFLIGHT smi /'
rc=0; python - <<'PY' || rc=$?
import torch, sys
from importlib.metadata import version, PackageNotFoundError
def v(p):
    try: return version(p)
    except PackageNotFoundError: return "absent"
print("PREFLIGHT torch", torch.__version__, "toolkit_cuda", torch.version.cuda,
      "driver_cuda", torch._C._cuda_getDriverVersion(), "cudnn", torch.backends.cudnn.version())
print("PREFLIGHT wheels cublas", v("nvidia-cublas-cu13"), "cudnn", v("nvidia-cudnn-cu13"),
      "cuda_runtime", v("nvidia-cuda-runtime-cu13"))
print("PREFLIGHT gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
ok = True
for dt in ("float32", "float16", "bfloat16"):
    try:
        a = torch.randn(1024, 1024, device="cuda", dtype=getattr(torch, dt)); (a @ a).float().sum().item()
        print(f"PREFLIGHT GEMM {dt}: OK")
    except Exception as e:
        ok = False; print(f"PREFLIGHT GEMM {dt}: FAIL {type(e).__name__}: {str(e)[:110]}")
sys.exit(0 if ok else 2)
PY
echo "== END PREFLIGHT =="
if [ "$rc" -ne 0 ]; then
  echo "PREFLIGHT: basic GEMM failed on this node -> CUDA-13/driver/cuBLAS mismatch (versions above), NOT the harness. Aborting before the 30GB model load."
  exit 1
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
fi
echo "DONE -> $OUTPUT_DIR (uri: ${OUTPUT_URI:-none})"
