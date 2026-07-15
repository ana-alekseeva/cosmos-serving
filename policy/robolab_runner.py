"""Real RoboLab rollout driver (Job 3) — Isaac box only.

Drives the RoboLab benchmark (NVLabs/RoboLab, Isaac Lab) against a DEPLOYED Cosmos3
endpoint and turns the per-task rollout outputs into the success dict the gate in
policy/robolab.py compares. The simulator and the policy never share a process: RoboLab's
sim-side client speaks the OpenPI websocket protocol, and vLLM-Omni serves exactly that
protocol at /v1/realtime/robot/openpi (endpoint deployed with a policy_server_config
stage override + cosmos_framework on the server's PYTHONPATH).

Contract, verified against the RoboLab docs (2026-07-15):
  * Server-client eval: the policy is a standalone server; a lightweight client inside
    the simulator sends observations and receives actions (docs/policy.md).
  * Default DROID observation registration — over_shoulder_left_camera + wrist_cam
    (N,H,W,3 uint8), arm_joint_pos (N,7), gripper_pos (N,1) — matches the Cosmos DROID
    contract (joint(7)+gripper(1), wrist + exterior views). The openpi packing seam is
    RoboLab's _pack_request(); VERIFY the served schema on-box against the server's
    policy_server_config handshake.
  * Runner: the pi0_family README documents `--policy ... --task <T> --num-envs N
    --headless --remote-uri <URI>` (--remote-uri overrides --remote-host/--remote-port —
    required here, because the Cosmos route lives at a PATH under the HTTP port, which
    host/port alone cannot express). docs/policy.md also shows a `run_eval.py` form;
    VERIFY which entrypoint the pinned ref ships (ROBOLAB_RUNNER selects it).
  * Results land under output/<timestamp or name>/ as JSON success metrics
    (docs/policy.md); the exact schema is unpublished, so parse_task_success() accepts
    the plausible shapes and fails loudly (listing every file it saw) otherwise.

Reruns are idempotent: each parsed task writes a per-task record under the rollout
dir, and an existing record short-circuits the subprocess — a crashed job resumes where
it stopped instead of re-simulating finished tasks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from policy.configs import Config
from policy.dataset import QualityTask, quality_subset

# vLLM-Omni's OpenPI websocket route (verified 0.24.0: vllm_omni/entrypoints/openpi/).
OPENPI_ROUTE = "/v1/realtime/robot/openpi"

# Runner entrypoint + client policy, relative to the RoboLab checkout. `pi05` selects
# RoboLab's openpi websocket client packing — the actual policy is the SERVER (Cosmos);
# the client flag only picks the observation packing. VERIFY both on-box.
DEFAULT_RUNNER = "policies/pi0_family/run.py"
DEFAULT_CLIENT_POLICY = "pi05"

TASK_MAP_PATH = Path(__file__).resolve().parent.parent / "config" / "robolab_tasks.yaml"

# Episodes run as parallel sim envs, every subset task caps at <=90 s, and the whole eval
# budgets <1 h per endpoint (two parallel L40S jobs) — so one task should take minutes.
# 15 min covers scene load + inference stalls while stopping a hung sim from eating the
# budget; ROBOLAB_TASK_TIMEOUT_S overrides for slow first-run asset downloads.
TASK_TIMEOUT_S = float(os.environ.get("ROBOLAB_TASK_TIMEOUT_S", 900))


def openpi_uri(endpoint: str) -> str:
    """Endpoint base URL -> the OpenPI websocket URI (http->ws, https->wss)."""
    s = urlsplit(endpoint if "//" in endpoint else "https://" + endpoint)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(s.scheme)
    if scheme is None:
        raise ValueError(f"unsupported endpoint scheme {s.scheme!r} in {endpoint!r}")
    path = s.path.rstrip("/")
    if not path.endswith(OPENPI_ROUTE):
        path += OPENPI_ROUTE
    return f"{scheme}://{s.netloc}{path}"


def load_task_map(path: str | Path | None = None, *,
                  subset: list[QualityTask] | None = None) -> dict[str, str]:
    """Load the slot->real-task-name map, refusing unfilled or unknown slots."""
    p = Path(path) if path else TASK_MAP_PATH
    raw = yaml.safe_load(p.read_text()) or {}
    subset = subset or quality_subset()
    slots = {t.task for t in subset}
    unknown = sorted(set(raw) - slots)
    if unknown:
        raise ValueError(f"{p}: unknown subset slots {unknown} — keys must match "
                         "policy/dataset.quality_subset() ids")
    missing = sorted(s for s in slots if not raw.get(s))
    if missing:
        raise ValueError(
            f"{p}: {len(missing)} unfilled slot(s) {missing} — pick real RoboLab task names "
            "per cell (generate the catalog with RoboLab scripts/generate_task_metadata.py, "
            "see the comments in that file) before running the real backend")
    return {s: str(raw[s]) for s in slots}


def rollout_cmd(robolab_root: Path, task_name: str, episodes: int, uri: str,
                run_name: str, *, python: str | None = None,
                runner: str | None = None, client_policy: str | None = None) -> list[str]:
    """The RoboLab runner invocation for one task (flags per the pi0_family README).

    Job-level env knobs (like TENSOR_PARALLEL_SIZE in policy/serving.py): ROBOLAB_PYTHON —
    the image's Isaac interpreter, NOT our uv venv (Isaac Lab lives in the kit env);
    ROBOLAB_RUNNER / ROBOLAB_CLIENT_POLICY override the entrypoint / client packing."""
    python = python or os.environ.get("ROBOLAB_PYTHON") or sys.executable
    runner = runner or os.environ.get("ROBOLAB_RUNNER") or DEFAULT_RUNNER
    client_policy = client_policy or os.environ.get("ROBOLAB_CLIENT_POLICY") or DEFAULT_CLIENT_POLICY
    return [python, str(Path(robolab_root) / runner),
            "--policy", client_policy,
            "--task", task_name,
            "--num-envs", str(episodes),
            "--headless",
            "--remote-uri", uri,
            "--output-folder-name", run_name]


# ---------------------------------------------------------------------------
# Rollout-output parsing. RoboLab's success-metrics JSON schema is unpublished; accept
# the plausible shapes (rate field / success+episode counters / per-episode lists) and
# fail loudly otherwise — never guess a number.
# ---------------------------------------------------------------------------
_NAME_HINTS = ("success", "summary", "metric", "result", "eval")
_COUNT_KEYS = ("episodes", "num_runs", "num_episodes", "trials", "num_trials")


def _success_from_list(v: list) -> tuple[float, int] | None:
    flags = []
    for item in v:
        if isinstance(item, bool):
            flags.append(item)
        elif isinstance(item, dict) and isinstance(item.get("success"), bool):
            flags.append(item["success"])
        else:
            return None
    return (sum(flags) / len(flags), len(flags)) if flags else None


def _success_from_obj(obj) -> tuple[float, int] | None:
    if isinstance(obj, list):
        return _success_from_list(obj)
    if not isinstance(obj, dict):
        return None
    count = next((int(obj[k]) for k in _COUNT_KEYS if isinstance(obj.get(k), int)), 0)
    for k in ("success_rate", "overall_success", "mean_success"):
        if isinstance(obj.get(k), (int, float)) and not isinstance(obj[k], bool):
            return float(obj[k]), count
    if isinstance(obj.get("successes"), int) and count:
        return obj["successes"] / count, count
    for k in ("episodes", "results", "runs"):
        if isinstance(obj.get(k), list):
            got = _success_from_list(obj[k])
            if got:
                return got
    return None


def parse_task_success(out_dir: str | Path) -> tuple[float, int]:
    """One rollout output dir -> (success_rate, episodes counted); name-hinted files first."""
    files = sorted(Path(out_dir).rglob("*.json"))
    ranked = sorted(files, key=lambda p: (not any(h in p.name.lower() for h in _NAME_HINTS),
                                          str(p)))
    for p in ranked:
        try:
            obj = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        got = _success_from_obj(obj)
        if got:
            return got
    raise RuntimeError(
        f"no parsable success metrics under {out_dir}; json files seen: "
        f"{[str(f) for f in files]} — inspect the RoboLab output schema (docs/policy.md) "
        "and extend policy/robolab_runner.parse_task_success")


def run_task(config: Config, qtask: QualityTask, task_name: str, *, robolab_root: Path,
             rollout_dir: Path, uri: str, episodes: int, python: str | None = None,
             runner: str | None = None, client_policy: str | None = None,
             timeout_s: float = TASK_TIMEOUT_S) -> dict:
    """Roll out one (config, task) and record its success; existing records short-circuit."""
    record_path = Path(rollout_dir) / config.cid / f"{qtask.task}.json"
    if record_path.exists():
        return json.loads(record_path.read_text())      # resume: never re-simulate
    # Only an actual rollout needs the checkout — the gate run resumes from records alone.
    if not Path(robolab_root).is_dir():
        raise FileNotFoundError(
            f"no record for {config.cid}/{qtask.task} and no RoboLab checkout at "
            f"{robolab_root} — clone NVLabs/RoboLab there (the Job 3 setup block does) or "
            "sync the missing rollout records into the rollout dir")
    record_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = f"{config.cid}_{task_name}"
    cmd = rollout_cmd(robolab_root, task_name, episodes, uri, run_name,
                      python=python, runner=runner, client_policy=client_policy)
    log_path = record_path.with_suffix(".log")
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n"); log.flush()
        # cwd = the checkout: RoboLab resolves assets + its output/ dir relative to root.
        proc = subprocess.run(cmd, cwd=robolab_root, stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(f"RoboLab rollout failed for {task_name} "
                           f"(exit {proc.returncode}) — see {log_path}")

    # --output-folder-name names the run under output/ (VERIFY exact layout on-box);
    # match loosely and take the newest so a timestamp prefix/suffix still resolves.
    out_dirs = sorted((Path(robolab_root) / "output").glob(f"*{run_name}*"),
                      key=lambda p: p.stat().st_mtime)
    if not out_dirs:
        raise FileNotFoundError(
            f"RoboLab wrote no output dir matching *{run_name}* under "
            f"{Path(robolab_root) / 'output'} — see {log_path} and the "
            "--output-folder-name VERIFY note in policy/robolab_runner.py")
    rate, counted = parse_task_success(out_dirs[-1])

    record = {"task": qtask.task, "robolab_task": task_name,
              "capability": qtask.capability, "difficulty": qtask.difficulty,
              "episodes": counted or episodes, "success_rate": round(rate, 4),
              "source": str(out_dirs[-1])}
    record_path.write_text(json.dumps(record, indent=2))
    return record


def run_quality_subset_real(config: Config, endpoint: str, subset: list[QualityTask], *,
                            robolab_root: str | Path, rollout_dir: str | Path,
                            episodes: int | None = None, python: str | None = None,
                            runner: str | None = None, client_policy: str | None = None,
                            task_map: dict[str, str] | None = None) -> dict:
    """Run the 18-task subset for one config against one endpoint -> the gate's dict
    (same shape as the mock's run_quality_subset, plus robolab bookkeeping)."""
    if not endpoint:
        raise ValueError(f"{config.cid}: the real RoboLab backend needs an endpoint URL "
                         "(--endpoint-baseline/--endpoint-candidate)")
    task_map = task_map or load_task_map(subset=subset)
    uri = openpi_uri(endpoint)
    per_task = []
    for t in subset:
        rec = run_task(config, t, task_map[t.task], robolab_root=Path(robolab_root),
                       rollout_dir=Path(rollout_dir), uri=uri,
                       episodes=episodes or t.episodes, python=python, runner=runner,
                       client_policy=client_policy)
        per_task.append({"task": t.task, "capability": t.capability,
                         "difficulty": t.difficulty, "episodes": rec["episodes"],
                         "success_rate": rec["success_rate"]})
    overall = round(sum(p["success_rate"] for p in per_task) / len(per_task), 4)
    return {"configuration": config.cid, "backend": "robolab", "endpoint": endpoint,
            "overall_success": overall, "per_task": per_task}
