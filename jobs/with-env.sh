#!/usr/bin/env bash
# Load the repo .env into the environment, then run the given command — so npa and SkyPilot
# fetch secrets/config from the environment automatically. npa reads os.environ (or
# ~/.npa/credentials.yaml); this feeds the environment path from a single .env, so you don't
# hand-export tokens before every launch.
#
#   ./jobs/with-env.sh npa workbench cosmos deploy --runtime serverless ...
#   ./jobs/with-env.sh sky jobs launch jobs/job1-ablation-matrix.sky.yaml
#   ./jobs/with-env.sh bash deploy/setup_gpu.sh
#
# Override the file with ENV_FILE=/path/to/other.env ./jobs/with-env.sh ...
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_file="${ENV_FILE:-$root/.env}"

if [ ! -f "$env_file" ]; then
  echo "with-env: missing $env_file — copy .env.example to .env and fill it in" >&2
  exit 1
fi
if [ "$#" -eq 0 ]; then
  echo "usage: jobs/with-env.sh <command> [args...]   (e.g. sky jobs launch <spec>)" >&2
  exit 2
fi

set -a            # export every variable defined while sourcing
# shellcheck disable=SC1090
. "$env_file"
set +a

exec "$@"
