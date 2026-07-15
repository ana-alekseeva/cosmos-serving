"""RoboLab quality evaluation: gates the lossy techniques on measured task success."""
from __future__ import annotations

import json
from pathlib import Path

from policy.config import CONFIG
from policy.configs import Config, config_by_id
from policy.dataset import QualityTask, quality_subset

# Reject if success drops by more than this (run parameter from config/experiment.yaml).
SUCCESS_DROP_THRESHOLD = CONFIG.quality_gate.robolab_success_drop


def run_quality_subset(config: Config, *, backend: str = "mock",
                       endpoint: str | None = None,
                       subset: list[QualityTask] | None = None, seed: int = 0,
                       robolab_root: Path | None = None,
                       rollout_dir: Path | None = None,
                       episodes: int | None = None) -> dict:
    """Run the 18-task subset for one configuration -> per-task + overall success."""
    subset = subset or quality_subset()
    if backend != "mock":
        return _run_real_robolab(config, endpoint, subset, robolab_root=robolab_root,
                                 rollout_dir=rollout_dir, episodes=episodes)
    from policy.mock.robolab import config_success
    per_task = []
    for t in subset:
        s = config_success(t, config, seed=seed)
        per_task.append({"task": t.task, "capability": t.capability,
                         "difficulty": t.difficulty, "episodes": t.episodes,
                         "success_rate": round(s, 4)})
    overall = round(sum(p["success_rate"] for p in per_task) / len(per_task), 4)
    return {"configuration": config.cid, "backend": backend,
            "overall_success": overall, "per_task": per_task}


def compare(baseline_cid: str, candidate_cid: str, *, backend: str = "mock",
            endpoint_baseline: str | None = None, endpoint_candidate: str | None = None,
            robolab_root: Path | None = None, rollout_dir: Path | None = None,
            episodes: int | None = None) -> dict:
    """Subset comparison + accept/reject vs SUCCESS_DROP_THRESHOLD (lossy gate)."""
    base = run_quality_subset(config_by_id(baseline_cid), backend=backend,
                              endpoint=endpoint_baseline, robolab_root=robolab_root,
                              rollout_dir=rollout_dir, episodes=episodes)
    cand = run_quality_subset(config_by_id(candidate_cid), backend=backend,
                              endpoint=endpoint_candidate, robolab_root=robolab_root,
                              rollout_dir=rollout_dir, episodes=episodes)
    drop = round(base["overall_success"] - cand["overall_success"], 4)
    return {
        "baseline": baseline_cid, "candidate": candidate_cid,
        "baseline_success": base["overall_success"],
        "candidate_success": cand["overall_success"],
        "success_drop": drop, "threshold": SUCCESS_DROP_THRESHOLD,
        "passed": drop <= SUCCESS_DROP_THRESHOLD,
        "baseline_detail": base, "candidate_detail": cand,
    }


def _run_real_robolab(config: Config, endpoint: str | None, subset, *,
                      robolab_root: Path | None = None,
                      rollout_dir: Path | None = None,
                      episodes: int | None = None) -> dict:
    """Real RoboLab eval (Job 3). Import stays local: the driver is Isaac-box-only."""
    from policy.robolab_runner import run_quality_subset_real

    if robolab_root is None:
        raise ValueError(
            f"{config.cid}: real RoboLab eval needs --robolab-root (the NVLabs/RoboLab "
            "checkout on the Isaac box; jobs/job3-robolab-subset.sky.yaml sets it). "
            "The mock backend needs no simulator.")
    return run_quality_subset_real(
        config, endpoint, subset, robolab_root=robolab_root,
        rollout_dir=rollout_dir or Path("results") / "robolab", episodes=episodes)


def write_report(result: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    return p
