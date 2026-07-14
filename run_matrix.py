#!/usr/bin/env python
"""PyTorch ablation matrix (specification_revised.txt §4 Job 1).

Runs every single-GPU configuration (R0-R4, G0-G5, E0-E6) as an isolated subprocess, with
the §8 bias controls (baseline at start + end, randomized order, wait between, drift
rejection). One provisioned GPU; each config releases its CUDA context on exit.

    python run_matrix.py --config config/experiment.yaml \
        --checkpoint-dir /local/model \
        --input-manifest /local/replay/manifest.json \
        --output-dir results

Validate the whole pipeline first with a fast smoke run — 1 request per configuration, no
warm-ups, no inter-config waits (each config still runs as its own subprocess, so this
exercises the real CLI + subprocess + logging path, just quickly):

    python run_matrix.py --smoke --output-dir results-smoke
    python aggregate.py --out-dir results-smoke
"""
from __future__ import annotations

from pathlib import Path

import typer

from policy.experiment import load_experiment
from policy.matrix import run_matrix

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Path = typer.Option(None, "--config", help="experiment.yaml (optional; code defaults otherwise)"),
    checkpoint_dir: str = typer.Option(None, "--checkpoint-dir"),
    input_manifest: str = typer.Option(None, "--input-manifest"),
    output_dir: str = typer.Option(None, "--output-dir"),
    backend: str = typer.Option(None, "--backend", help="mock | vllm"),
    endpoint: str = typer.Option(None, "--endpoint"),
    configurations: str = typer.Option(None, "--configurations", help="comma-separated cids (default: full matrix)"),
    smoke: bool = typer.Option(False, "--smoke", help="fast validation: 1 request/config, 0 warm-ups, no waits"),
    replay_size: int = typer.Option(None, "--replay-size", help="measured requests per config (override)"),
    warmups: int = typer.Option(None, "--warmups", help="warm-up requests per config (override)"),
    no_subprocess: bool = typer.Option(False, "--no-subprocess", help="run in-process (no CUDA isolation)"),
) -> None:
    if smoke:                                    # 1 request/config, no warm-ups, no inter-config waits
        replay_size = 1 if replay_size is None else replay_size
        warmups = 0 if warmups is None else warmups
    exp = load_experiment(config).override(
        checkpoint_dir=checkpoint_dir, input_manifest=input_manifest,
        output_dir=output_dir, backend=backend, endpoint=endpoint,
        replay_size=replay_size, warmup_requests=warmups,
        wait_between_seconds=0.0 if smoke else None,
        configurations=[c.strip() for c in configurations.split(",")] if configurations else None,
    )
    if smoke:
        typer.echo(f"SMOKE RUN — {exp.replay_size} request/config, {exp.warmup_requests} warm-ups, "
                   f"backend={exp.backend}. Validates the full matrix→logs pipeline quickly.\n")
    status = run_matrix(exp, spawn=not no_subprocess)
    typer.echo(f"\nmatrix done — ran {len(status['configurations_run'])} configs; "
               f"{len(status['failed'])} failed; "
               f"rejected={status['rejected']} (see {exp.output_dir}/matrix_status.json)")
    typer.echo(f"next: python aggregate.py --out-dir {exp.output_dir}")
    if status["failed"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
