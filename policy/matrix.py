"""PyTorch ablation matrix orchestrator (specification_revised.txt §4 Job 1, §8).

Runs every single-GPU configuration (R0-R4, G0-G5, E0-E6) as an isolated subprocess so
each releases its CUDA context before the next starts (§4). Applies the §8 bias controls
for one long provisioned job:

    Baseline (E0)
      -> configuration matrix in randomized order
      -> Baseline (E0) repeated

Then compares the two baseline measurements and REJECTS the run if they differ beyond
`baseline_drift_reject_pct` (GPU drift). Waits briefly between subprocesses; stages inputs
locally first. This module is the importable core; run_matrix.py is the CLI wrapper.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

from policy.configs import END_TO_END, baseline_id
from policy.experiment import Experiment
from policy.runner import load_requests, resolve_configs, run_configuration

REPO_ROOT = Path(__file__).resolve().parent.parent


def _drift_pct(a: float, b: float) -> float:
    return abs(a - b) / a * 100.0 if a else float("inf")


def _p50_chunk(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text())
    return data.get("percentiles", {}).get("total_chunk_ms", {}).get("p50")


def _spawn_one(cid: str, exp: Experiment, out_subdir: str, is_baseline: bool) -> None:
    """Run one configuration in its own process (releases CUDA context on exit, §4)."""
    cmd = [
        sys.executable, str(REPO_ROOT / "run_configuration.py"),
        "--configuration", cid,
        "--backend", exp.backend,
        "--out-dir", exp.output_dir,
        "--out-subdir", out_subdir,
        "--run-id", exp.run_id,
        "--model", exp.model,
        "--manifest", exp.input_manifest,
        "--warmups", str(exp.warmup_requests),
        "--torchinductor-root", exp.torchinductor_root,
        "--replay-size", str(exp.replay_size),
    ]
    if exp.endpoint:
        cmd += ["--endpoint", exp.endpoint]
    if is_baseline:
        cmd += ["--is-baseline"]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def run_matrix(exp: Experiment, *, spawn: bool = True) -> dict:
    """Run the full ablation matrix with §8 bias controls. Returns the matrix status dict."""
    requests = load_requests(exp)                       # stage inputs locally before timing (§8)
    configs = resolve_configs(exp.configurations)
    out_dir = Path(exp.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cid = baseline_id(END_TO_END)                  # E0 — the combined-pipeline baseline
    have_base = any(c.cid == base_cid for c in configs)

    # Order (§8): baseline first, matrix randomized, baseline repeated.
    middle = [c.cid for c in configs]
    if exp.randomize_order:
        random.Random(exp.order_seed).shuffle(middle)
    plan: list[tuple[str, str, bool]] = []              # (cid, out_subdir, is_baseline)
    if exp.baseline_at_start_and_end and have_base:
        plan.append((base_cid, base_cid, True))
        middle = [c for c in middle if c != base_cid]   # already scheduled at the start
    plan += [(cid, cid, cid == base_cid) for cid in middle]
    if exp.baseline_at_start_and_end and have_base:
        plan.append((base_cid, f"{base_cid}_end", True))

    ran, failed = [], []
    for i, (cid, subdir, is_base) in enumerate(plan):
        print(f"» [{i + 1}/{len(plan)}] {cid}"
              f"{' (baseline)' if is_base else ''} -> {out_dir / subdir}", flush=True)
        try:
            if spawn:
                _spawn_one(cid, exp, subdir, is_base)
            else:                                       # in-process (tests / no subprocess)
                from policy.configs import config_by_id
                run_configuration(config_by_id(cid), requests, backend=exp.backend,
                                  out_dir=exp.output_dir, run_id=exp.run_id, model=exp.model,
                                  endpoint=exp.endpoint, warmups=exp.warmup_requests,
                                  is_baseline=is_base, torchinductor_root=exp.torchinductor_root,
                                  out_subdir=subdir)
            ran.append(subdir)
        except Exception as exc:
            failed.append((cid, str(exc)[:400]))
            print(f"  !! {cid} FAILED: {str(exc)[:200]}", flush=True)
        if i < len(plan) - 1 and exp.wait_between_seconds:
            time.sleep(exp.wait_between_seconds)        # let the GPU settle between configs (§8)

    status = _finish(exp, out_dir, base_cid, have_base, ran, failed)
    (out_dir / "matrix_status.json").write_text(json.dumps(status, indent=2))
    return status


def _finish(exp, out_dir, base_cid, have_base, ran, failed) -> dict:
    drift = None
    rejected = False
    if exp.baseline_at_start_and_end and have_base:
        a = _p50_chunk(out_dir / base_cid / "summary.json")
        b = _p50_chunk(out_dir / f"{base_cid}_end" / "summary.json")
        if a is not None and b is not None:
            d = _drift_pct(a, b)
            rejected = d > exp.baseline_drift_reject_pct
            drift = {"baseline_start_p50_ms": a, "baseline_end_p50_ms": b,
                     "drift_pct": round(d, 2), "reject_threshold_pct": exp.baseline_drift_reject_pct,
                     "rejected": rejected}
            print(f"\nbaseline drift ({base_cid} start vs end): {d:.2f}% "
                  f"-> {'REJECT' if rejected else 'accept'}", flush=True)
    return {
        "run_id": exp.run_id, "backend": exp.backend, "output_dir": str(out_dir),
        "configurations_run": ran, "failed": failed,
        "baseline_drift": drift, "rejected": rejected,
    }
