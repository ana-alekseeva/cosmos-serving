"""RoboLab quality evaluation (specification_revised.txt §4 Jobs 3-4, §5).

The waterfalls optimize *latency*; RoboLab confirms the lossy techniques (Cache-DiT, FP8)
don't damage *policy performance*, measured as task success from executing the generated
actions — not a pixel proxy (§1: "evaluated by executing its generated actions in RoboLab").

  * Job 3 — subset: the stratified 18-task subset (§5), 10 episodes each, for
    {baseline, final optimized PyTorch, final production}. Rejects an optimization that
    regresses success beyond the threshold.
  * Job 4 — full: the complete RoboLab benchmark for {baseline, final production} only,
    run after the subset passes.

The mock models per-task success so the gate logic + report render with no simulator. The
real path drives RoboLab against a deployed endpoint (see jobs/robolab-eval.*).
"""
from __future__ import annotations

import json
from pathlib import Path

from policy.config import CONFIG
from policy.configs import Config, config_by_id
from policy.dataset import QualityTask, quality_subset

# Success-drop gate threshold — a run parameter from the single config file
# (config/experiment.yaml -> quality_gate). Reject if success drops by more than this.
SUCCESS_DROP_THRESHOLD = CONFIG.quality_gate.robolab_success_drop


def run_quality_subset(config: Config, *, backend: str = "mock",
                       endpoint: str | None = None,
                       subset: list[QualityTask] | None = None, seed: int = 0) -> dict:
    """Run the 18-task subset for one configuration -> per-task + overall success."""
    subset = subset or quality_subset()
    if backend != "mock":
        return _run_real_robolab(config, endpoint, subset)      # VERIFY: on the RT-core box
    from policy.mock.robolab import config_success              # modeled success (no simulator)
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
            endpoint_baseline: str | None = None, endpoint_candidate: str | None = None) -> dict:
    """Subset comparison + accept/reject vs SUCCESS_DROP_THRESHOLD (§9 lossy gate)."""
    base = run_quality_subset(config_by_id(baseline_cid), backend=backend,
                              endpoint=endpoint_baseline)
    cand = run_quality_subset(config_by_id(candidate_cid), backend=backend,
                              endpoint=endpoint_candidate)
    drop = round(base["overall_success"] - cand["overall_success"], 4)
    return {
        "baseline": baseline_cid, "candidate": candidate_cid,
        "baseline_success": base["overall_success"],
        "candidate_success": cand["overall_success"],
        "success_drop": drop, "threshold": SUCCESS_DROP_THRESHOLD,
        "passed": drop <= SUCCESS_DROP_THRESHOLD,
        "baseline_detail": base, "candidate_detail": cand,
    }


def _run_real_robolab(config: Config, endpoint: str | None, subset) -> dict:
    raise NotImplementedError(
        "Real RoboLab eval runs on an RT-core GPU (Isaac Sim/Isaac Lab), not the H200 "
        "vLLM-Omni image (§4 Job 3). Drive it via jobs/robolab-eval.*; the mock models "
        "success offline."
    )


def write_report(result: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2))
    return p
