#!/usr/bin/env python3
"""Standing-endpoint launcher for cosmos-droid-vllm (Path A: npa serverless).

Builds the exact `vllm-omni serve` command for one E-config and execs it, turning the
container into a long-lived policy server. npa serverless runs the image ENTRYPOINT and
injects ONLY env (COSMOS_MODEL_ID, COSMOS_SERVER_PORT) — never a command — so the image
must start the server itself.

ONE process exposes BOTH eval routes (Topology note in policy/serving.py):
  * POST /v1/videos               — latency harness (run_matrix.py --endpoint, Job 2)
  * ws   /v1/realtime/robot/openpi — RoboLab rollouts (Job 3)

The OpenPI websocket route is mounted ONLY when the diffusion stage carries a
policy_server_config, so we pass it via --stage-overrides. policy/serving.py::start_policy_server
omits it because Job 2 latency only needs /v1/videos; here we add it for RoboLab.
"""
from __future__ import annotations

import argparse
import json
import os

from policy.configs import config_by_id
from policy.serving import DEFAULT_MODEL, OMNI_FLAG, SERVE_CMD, engine_args

# DROID observation contract: wrist + 2 external cameras, joint-position action space.
# Verified block from deploy/serve_e0_vm.sh — the diffusion stage (index "0") reads it and
# mounts /v1/realtime/robot/openpi. Keep in lockstep with the DROID payload in
# policy/serving.py::build_request_parts (2 exterior views + wrist -> composed input_reference).
STAGE_OVERRIDES = {
    "0": {
        "model_config": {
            "policy_server_config": {
                "image_resolution": [540, 640],
                "n_external_cameras": 2,
                "needs_wrist_camera": True,
                "needs_stereo_camera": False,
                "needs_session_id": True,
                "action_space": "joint_position",
            }
        }
    }
}


def build_cmd(model: str, cid: str, host: str, port: int) -> list[str]:
    """`vllm-omni serve <model> --omni <engine_args(cid)> --stage-overrides <policy_server_config>`."""
    config = config_by_id(cid)  # "E0".."E4" -> cumulative engine flags (attention/compile/graphs/fp8)
    return [
        *SERVE_CMD, model, OMNI_FLAG,
        *engine_args(config),
        "--host", host, "--port", str(port),
        "--stage-overrides", json.dumps(STAGE_OVERRIDES),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("COSMOS_MODEL_ID", DEFAULT_MODEL))
    ap.add_argument("--config", default=os.environ.get("CONFIG", "E0"),
                    help="E-ladder rung: E0 (baseline eager) .. E4 (FP8, final).")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("COSMOS_SERVER_PORT", "8080")))
    a = ap.parse_args()

    cmd = build_cmd(a.model, a.config, a.host, a.port)
    print("serve:", " ".join(cmd), flush=True)
    os.execvp(cmd[0], cmd)  # replace this process with the server (becomes container PID 1's child)


if __name__ == "__main__":
    main()
