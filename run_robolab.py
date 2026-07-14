#!/usr/bin/env python
"""RoboLab quality evaluation (specification_revised.txt §4 Jobs 3-4, §5).

Compares a candidate configuration's task success against the baseline on the stratified
18-task subset (Job 3) and rejects it if success regresses beyond the threshold — the gate
for the lossy Cache-DiT / FP8 techniques. Run the full benchmark (Job 4) only after the
subset passes.

    python run_robolab.py --baseline E0 --candidate E6 --out-dir results
"""
from __future__ import annotations

from pathlib import Path

import typer

from policy.robolab import compare, write_report

app = typer.Typer(add_completion=False)


@app.command()
def main(
    baseline: str = typer.Option("E0", "--baseline", help="baseline config id"),
    candidate: str = typer.Option("E6", "--candidate", help="candidate (final optimized) config id"),
    backend: str = typer.Option("mock", "--backend", help="mock | vllm"),
    endpoint_baseline: str = typer.Option(None, "--endpoint-baseline"),
    endpoint_candidate: str = typer.Option(None, "--endpoint-candidate"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
) -> None:
    result = compare(baseline, candidate, backend=backend,
                     endpoint_baseline=endpoint_baseline, endpoint_candidate=endpoint_candidate)
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
