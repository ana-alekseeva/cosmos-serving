#!/usr/bin/env bash
# RoboLab infra smoke (Job 3 prerequisite): validates the three risky unknowns cheaply —
# RT-core GPU quota, the Isaac Lab NGC image, and RoboLab repo access. No rollouts, no endpoint.
set -uo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "== Isaac Sim / Isaac Lab =="
python -c "import isaaclab; print('isaaclab', getattr(isaaclab, '__version__', 'ok'))" 2>&1 || \
  python -c "import omni.isaac.lab as il; print('isaac lab (omni):', 'ok')" 2>&1 || \
  echo "isaaclab import failed — check the image tag / python env"

echo "== RoboLab repo =="
if git clone --depth 1 https://github.com/NVLabs/RoboLab.git /tmp/rl 2>&1; then
  echo "--- top-level:"; ls /tmp/rl
  echo "--- candidate runner entrypoints:"
  find /tmp/rl -maxdepth 3 -name "*.py" | grep -iE "run|eval|rollout|policy" | head -20
  find /tmp/rl -maxdepth 2 -name "README*" -exec head -60 {} \;
else
  echo "RoboLab clone FAILED — repo private or URL wrong (# VERIFY in jobs/job3 yaml)"
fi
echo "== SMOKE DONE =="
