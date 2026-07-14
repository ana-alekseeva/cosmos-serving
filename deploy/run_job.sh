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
: "${HF_TOKEN:?set HF_TOKEN (nebius --env-secret HF_TOKEN=...)}"
: "${BACKEND:=pytorch}"; : "${MODE:=matrix}"; : "${CONFIGS:=}"
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
# shellcheck disable=SC1091
source "$FRAMEWORK/.venv/bin/activate"
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
