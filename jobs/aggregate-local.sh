#!/usr/bin/env bash
# Aggregation, run LOCALLY. Replaces the retired
# jobs/job5-aggregate.sky.yaml: aggregation is CPU-only (pandas/pyarrow/matplotlib over the
# latency logs), so it needs no cloud node at all.
#
# Pulls the raw per-configuration results from object storage, re-runs aggregate.py over the
# MERGED tree (Job 1 native + Job 2 production + RoboLab, whatever has landed), and uploads
# the regenerated aggregate back. Jobs 1/2 already aggregate their own slice inline; this is
# the cross-job re-aggregation (or figure regeneration) pass.
#
#   ./jobs/aggregate-local.sh                      # .env supplies AWS creds + OUTPUT_URI
#   NO_UPLOAD=1 ./jobs/aggregate-local.sh          # inspect results/aggregate/ locally only
#   INPUT_URIS="s3://…/raw/" ./jobs/aggregate-local.sh   # override which raw trees to pull
#
# Env (all optional, .env is sourced first):
#   OUTPUT_URI    base results URI (default from .env); raw trees + aggregate/ hang off it
#   INPUT_URIS    space-separated raw trees to merge (default: ${OUTPUT_URI}raw/ +
#                 ${OUTPUT_URI}production/raw/ — a missing tree is skipped with a warning)
#   RESULTS_DIR   local working dir (default results)
#   NO_UPLOAD=1   skip the upload of results/aggregate/ back to ${OUTPUT_URI}aggregate/
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [ -f "$root/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$root/.env"
  set +a
fi

: "${OUTPUT_URI:=s3://serverless-challenge/cosmos3-ablation-results/}"
: "${AWS_ENDPOINT_URL:=https://storage.eu-north1.nebius.cloud}"
# `=` not `:=` — an explicit INPUT_URIS="" means "aggregate $RESULTS_DIR as-is, pull nothing"
: "${INPUT_URIS=${OUTPUT_URI}raw/ ${OUTPUT_URI}production/raw/}"
: "${RESULTS_DIR:=results}"
mkdir -p "$RESULTS_DIR"

if command -v aws >/dev/null 2>&1 && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  for uri in $INPUT_URIS; do
    echo "aggregate-local: pulling $uri"
    aws s3 cp "$uri" "$RESULTS_DIR/" --recursive --endpoint-url "$AWS_ENDPOINT_URL" \
      || echo "aggregate-local: WARN nothing at $uri (job not run yet?) — skipping" >&2
  done
else
  echo "aggregate-local: no aws CLI or AWS creds — aggregating what's already in $RESULTS_DIR/" >&2
fi

# --extra aggregate = pandas + pyarrow for the Parquet summaries (CSV/figures work without).
uv run --extra aggregate python aggregate.py --out-dir "$RESULTS_DIR"

if [ -z "${NO_UPLOAD:-}" ] && command -v aws >/dev/null 2>&1 && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  aws s3 cp "$RESULTS_DIR/aggregate/" "${OUTPUT_URI}aggregate/" --recursive \
    --endpoint-url "$AWS_ENDPOINT_URL"
  echo "aggregate-local: aggregate -> ${OUTPUT_URI}aggregate/"
else
  echo "aggregate-local: upload skipped — output in $RESULTS_DIR/aggregate/"
fi
