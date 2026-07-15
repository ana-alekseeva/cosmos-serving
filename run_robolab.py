#!/usr/bin/env python
"""RoboLab quality evaluation.

Compares a candidate configuration's task success against the baseline on the stratified
18-task subset (Job 3) and rejects it if success regresses beyond the threshold — the gate
for the lossy Cache-DiT / FP8 techniques. Run the full benchmark (Job 4) only after the
subset passes.

    # gate-plumbing check, no simulator (candidate defaults to the ladder's final rung):
    python run_robolab.py --baseline E0 --out-dir results
    # real, two parallel L40S jobs (one endpoint each — halves the <1 h wall budget):
    python run_robolab.py --backend vllm --robolab-root RoboLab --side baseline \
        --endpoint-baseline https://<baseline>
    python run_robolab.py --backend vllm --robolab-root RoboLab --side candidate \
        --endpoint-candidate https://<optimized>
    # then gate from the merged per-task records (resumes; no simulator needed):
    python run_robolab.py --backend vllm --robolab-root RoboLab --side both \
        --endpoint-baseline https://<baseline> --endpoint-candidate https://<optimized>
"""
from __future__ import annotations

from pathlib import Path

import typer

from policy.configs import END_TO_END_LADDER, config_by_id
from policy.robolab import compare, run_quality_subset, write_report

app = typer.Typer(add_completion=False)


@app.command()
def main(
    baseline: str = typer.Option("E0", "--baseline", help="baseline config id"),
    candidate: str = typer.Option(None, "--candidate",
                                  help="candidate config id (default: the ladder's final rung)"),
    backend: str = typer.Option("mock", "--backend", help="mock | vllm (vllm = real RoboLab rollouts)"),
    endpoint_baseline: str = typer.Option(None, "--endpoint-baseline", envvar="COSMOS_ENDPOINT_BASELINE"),
    endpoint_candidate: str = typer.Option(None, "--endpoint-candidate", envvar="COSMOS_ENDPOINT_OPTIMIZED"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    robolab_root: Path = typer.Option(None, "--robolab-root", envvar="ROBOLAB_ROOT",
                                      help="NVLabs/RoboLab checkout (real backend only)"),
    rollout_dir: Path = typer.Option(None, "--rollout-dir", envvar="ROBOLAB_ROLLOUT_DIR",
                                     help="per-task rollout records (resumable); default <out-dir>/robolab"),
    episodes: int = typer.Option(None, "--episodes", envvar="EPISODES_PER_TASK",
                                 help="episodes per task (default: subset spec, 10)"),
    side: str = typer.Option("both", "--side", envvar="ROBOLAB_SIDE",
                             help="baseline | candidate | both — one endpoint per job (two "
                                  "parallel L40S jobs); 'both' also gates, resuming from records"),
) -> None:
    if side not in {"baseline", "candidate", "both"}:
        raise typer.BadParameter(f"--side must be baseline|candidate|both, got {side!r}")
    candidate = candidate or END_TO_END_LADDER[-1].cid   # final optimized rung, whatever it is
    rollout_dir = rollout_dir or out_dir / "robolab"

    if side != "both":
        # One side of the comparison only (parallel L40S jobs). No gate yet: per-task
        # records land under rollout_dir/<cid>/ and the gate run merges + resumes them.
        cid, ep = ((baseline, endpoint_baseline) if side == "baseline"
                   else (candidate, endpoint_candidate))
        result = run_quality_subset(config_by_id(cid), backend=backend, endpoint=ep,
                                    robolab_root=robolab_root, rollout_dir=rollout_dir,
                                    episodes=episodes)
        path = write_report(result, out_dir / "aggregate" / f"robolab_subset_{side}.json")
        typer.echo(f"RoboLab subset [{side}] {cid}: success={result['overall_success']:.3f} "
                   f"over {len(result['per_task'])} tasks")
        typer.echo(f"report -> {path} — gate once both sides finish: sync the records and "
                   "rerun with --side both")
        return

    result = compare(baseline, candidate, backend=backend,
                     endpoint_baseline=endpoint_baseline, endpoint_candidate=endpoint_candidate,
                     robolab_root=robolab_root,
                     rollout_dir=rollout_dir, episodes=episodes)
    path = write_report(result, out_dir / "aggregate" / "robolab_subset.json")
    verdict = "PASS" if result["passed"] else "REJECT"
    typer.echo(f"RoboLab subset: baseline {baseline}={result['baseline_success']:.3f}  "
               f"candidate {candidate}={result['candidate_success']:.3f}  "
               f"drop={result['success_drop']:.3f} (thr {result['threshold']}) -> {verdict}")
    typer.echo(f"report -> {path}")
    if not result["passed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
