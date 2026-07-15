#!/usr/bin/env python
"""Separate multi-GPU experiment (specification_revised.txt §3).

Runs CFG-Parallel (if the action pipeline uses CFG) and Ulysses Context-Parallel on 2 GPUs
and compares each against the best single-GPU configuration (final E4). Kept OFF the primary
waterfall — "Do not mix different GPU counts in the primary waterfall" (§3).

    python run_multigpu.py --backend mock --out-dir results
"""
from __future__ import annotations

from pathlib import Path

import typer

from policy.multigpu import run_multigpu

app = typer.Typer(add_completion=False)


@app.command()
def main(
    backend: str = typer.Option("mock", "--backend", help="mock | vllm"),
    manifest: str = typer.Option("policy/mock/manifest.json", "--manifest"),
    model: str = typer.Option("nvidia/Cosmos3-Nano-Policy-DROID", "--model"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
) -> None:
    result = run_multigpu(backend=backend, manifest=manifest, model=model, out_dir=out_dir)
    typer.echo(f"best single-GPU ({result['best_single_gpu_config']}): "
               f"p50 chunk = {result['single_gpu_p50_chunk_ms']:.1f} ms")
    for row in result["strategies"]:
        if row.get("skipped"):
            typer.echo(f"  {row['strategy']}: skipped ({row['reason']})")
        else:
            typer.echo(f"  {row['label']}: {row['multigpu_p50_chunk_ms']:.1f} ms "
                       f"({row['speedup_vs_best_single_gpu']:.2f}× vs best single-GPU)")


if __name__ == "__main__":
    app()
