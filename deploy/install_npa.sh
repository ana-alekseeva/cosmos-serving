#!/usr/bin/env bash
# Install the Nebius Physical AI workbench (npa) into THIS project's environment.
#
# npa must be installed EDITABLE from a source checkout: the non-editable wheel install
# (`uv pip install "git+…#subdirectory=npa"`) crashes at startup because npa reads its
# `[tool.npa.supported-tools]` table from pyproject.toml, which isn't shipped in the wheel.
# So we clone into a project-local, gitignored dir and `uv pip install -e` from there.
#
# NOTE: npa is not in this project's pyproject/uv.lock (its editable path is machine-specific),
# so `uv sync` will PRUNE it from the venv — just re-run this script afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

SRC=".nebius-physical-ai"        # gitignored (see .gitignore)
REF="${NPA_REF:-main}"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/nebius/nebius-physical-ai.git "$SRC"
else
  git -C "$SRC" pull --ff-only || true
fi

uv pip install -e "$SRC/npa"
npa --version

cat <<'EON'

npa installed into the project venv. Next:
  npa configure --interactive     # ~/.npa/{credentials,config}.yaml: project/tenant/region,
                                  # S3 bucket + key, HF_TOKEN. Needs the "AI Jobs" IAM role.
Then create resources on Nebius with the real workbench commands — see jobs/README.md:
  bash jobs/deploy-optimized.sh   # npa workbench cosmos deploy --runtime serverless ...
EON
