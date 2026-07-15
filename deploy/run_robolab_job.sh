#!/usr/bin/env bash
# Raw Nebius AI Job entrypoint for Job 3 (RoboLab subset) — the no-SkyPilot path. Mirrors the
# setup+run of jobs/job3-robolab-subset.sky.yaml so it runs under `nebius ai job create` on an
# RT-core L40S. ONE endpoint per job (ROLE); run two jobs in parallel (baseline + optimized),
# then gate the merged records (run_robolab.py --side both) once both finish.
#
# Launch (per role) — inject this script + the local .env (secrets), pass the endpoint by env:
#   nebius ai job create --parent-id project-e00em6gppr002a5efwp7eb --name robolab-baseline \
#     --image nvcr.io/nvidia/isaac-lab:2.3.2 --platform gpu-l40s-a --preset 1gpu-32vcpu-128gb \
#     # gpu-l40s-a = Intel Ice Lake L40S (gpu-l40s-d is AMD Epyc — not available to this tenant)
#     --inject-file deploy/run_robolab_job.sh:/run_robolab_job.sh --inject-file .env:/work/.env \
#     --env ROLE=baseline --env COSMOS_ENDPOINT_BASELINE=https://<e0-url> \
#     --container-command bash --args /run_robolab_job.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

: "${REPO_URL:=https://github.com/ana-alekseeva/cosmos-serving.git}"
: "${REPO_REF:=main}"
: "${ROBOLAB_REPO:=https://github.com/NVLabs/RoboLab.git}"   # VERIFY ref + access
: "${ROBOLAB_REF:=main}"
: "${ROLE:=baseline}"
: "${EPISODES_PER_TASK:=10}"                 # run_robolab reads this via envvar; 10 = full subset
: "${SUCCESS_DROP_THRESHOLD:=0.03}"
# Job 3 uploads to OUTPUT_URI_EVAL, NOT OUTPUT_URI — .env's OUTPUT_URI is the Job 2 traces
# destination (cosmos3-ablation-results/); using it here would mix eval records into Job 2's tree.
: "${OUTPUT_URI_EVAL:=s3://serverless-challenge/robolab-eval-results/subset/}"
: "${AWS_ENDPOINT_URL:=https://storage.eu-north1.nebius.cloud}"
: "${ROBOLAB_PYTHON:=python}"                 # VERIFY: the image's Isaac interpreter (isaaclab.sh -p vs python)

WORK=/work; mkdir -p "$WORK"; cd "$WORK"
rm -rf cosmos-serving && git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" cosmos-serving
# Injected .env carries the secrets (HF_TOKEN, AWS_*). Copy it in and source it (never printed).
[ -f "$WORK/.env" ] && cp "$WORK/.env" cosmos-serving/.env
cd cosmos-serving
[ -f .env ] && { set -a; . ./.env; set +a; }
: "${HF_TOKEN:?inject .env with HF_TOKEN (Cosmos3-Nano-Policy-DROID license accepted on HF)}"
export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
uv sync

# RoboLab installed into the IMAGE's Isaac python (kit env), not our uv venv: the runner
# subprocess uses ROBOLAB_PYTHON. VERIFY on-box that pip -e resolves against the preinstalled
# Isaac Lab and that first-run asset download succeeds.
rm -rf RoboLab && git clone --depth 1 --branch "$ROBOLAB_REF" "$ROBOLAB_REPO" RoboLab
"${ROBOLAB_PYTHON}" -m pip install -e ./RoboLab

nvidia-smi || true
case "$ROLE" in
  baseline)  SIDE=baseline;  : "${COSMOS_ENDPOINT_BASELINE:?deploy the E0 endpoint first (MODE=baseline)}"  ;;
  optimized) SIDE=candidate; : "${COSMOS_ENDPOINT_OPTIMIZED:?deploy the E4 endpoint first (MODE=optimized)}" ;;
  *) echo "ROLE must be baseline|optimized, got '$ROLE'"; exit 1 ;;
esac
export ROBOLAB_ROOT="$PWD/RoboLab" ROBOLAB_PYTHON
echo "ROLE=$ROLE SIDE=$SIDE episodes=$EPISODES_PER_TASK threshold=$SUCCESS_DROP_THRESHOLD"

# Real rollouts. Per-task records under results/robolab/<cid>/ make reruns resume instead of
# re-simulating, so a relaunch after a crash is safe (does not restart the whole 18-task set).
uv run python run_robolab.py --backend vllm --side "$SIDE" --out-dir results

if command -v aws >/dev/null 2>&1 && [ -d results ]; then
  aws s3 cp results/ "${OUTPUT_URI_EVAL}raw/" --recursive --exclude "aggregate/*" --endpoint-url "$AWS_ENDPOINT_URL" || true
  [ -d results/aggregate ] && aws s3 cp results/aggregate/ "${OUTPUT_URI_EVAL}" --recursive --endpoint-url "$AWS_ENDPOINT_URL" || true
fi
echo "DONE role=$ROLE -> ${OUTPUT_URI_EVAL}raw/robolab/"
