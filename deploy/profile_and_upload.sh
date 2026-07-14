#!/usr/bin/env bash
# Profile a few native-PyTorch configs under torch.profiler and upload the Chrome traces to
# object storage, so the Perfetto latency dashboard can be built locally from S3.
#
# Self-guarding: no-ops (exit 0) unless the native-PyTorch stack is actually present — the
# cosmos_framework package is importable AND the checkpoint is staged. So it is safe to call
# from ANY job; the vLLM / RoboLab jobs simply skip it.
#
#   OUTPUT_URI / AWS_ENDPOINT_URL      — S3 destination (traces -> ${OUTPUT_URI}raw/traces/)
#   PROFILE_CONFIGS (default "R0 R1")  — configs to profile (eager vs +compile shows the split)
#   CHECKPOINT_DIR  (default /local/model), $1 = manifest (default /local/replay/manifest.json)
set -uo pipefail

MANIFEST="${1:-/local/replay/manifest.json}"
CKPT="${CHECKPOINT_DIR:-/local/model}"
CONFIGS="${PROFILE_CONFIGS:-R0 R1}"

# The profiler needs the native-PyTorch stack. If cosmos_framework isn't in the active env,
# try the sibling cosmos-framework/.venv that deploy/setup.sh sets up.
if ! python -c "import cosmos_framework" >/dev/null 2>&1; then
  cosmos_venv="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)/cosmos-framework/.venv/bin/activate"
  # shellcheck disable=SC1090
  [ -f "$cosmos_venv" ] && { set +u; source "$cosmos_venv"; set -u; }
fi
if ! python -c "import cosmos_framework" >/dev/null 2>&1; then
  echo "profile: cosmos_framework not importable — skipping torch.profiler traces" >&2
  exit 0
fi
if [ ! -d "$CKPT" ] || [ ! -f "$MANIFEST" ]; then
  echo "profile: no checkpoint ($CKPT) or manifest ($MANIFEST) — skipping traces" >&2
  exit 0
fi

mkdir -p traces
for c in $CONFIGS; do
  echo "profile: torch.profiler on $c ..."
  python -m policy.profile_pytorch --configuration "$c" --manifest "$MANIFEST" \
    --checkpoint-dir "$CKPT" --out "traces/trace_${c}.json" || echo "profile: $c failed" >&2
done

if command -v aws >/dev/null 2>&1 && [ -n "${OUTPUT_URI:-}" ]; then
  aws s3 cp traces/ "${OUTPUT_URI}raw/traces/" --recursive --endpoint-url "${AWS_ENDPOINT_URL}"
  echo "profile: traces -> ${OUTPUT_URI}raw/traces/  (download + open at ui.perfetto.dev)"
else
  echo "profile: traces in $(pwd)/traces (no aws / OUTPUT_URI to upload)" >&2
fi
