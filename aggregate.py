#!/usr/bin/env python
"""Aggregation (specification_revised.txt §4 Job 5).

A small CPU job that merges every per-configuration output and generates CSV/Parquet
summaries, the three waterfalls (reasoner/generator/end-to-end), the baseline-vs-final
stage breakdown, confidence intervals, quality-comparison tables, and figures.

    python aggregate.py --out-dir results
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from policy.aggregate import aggregate

app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="matrix output dir"),
    no_plots: bool = typer.Option(False, "--no-plots"),
) -> None:
    manifest = aggregate(out_dir, make_plots=not no_plots)
    typer.echo(f"aggregated {len(manifest['configurations'])} configs -> {manifest['output_dir']}")
    typer.echo(f"waterfalls: {', '.join(manifest['waterfalls'])}")
    typer.echo(f"CSV: {manifest['csv']}" + (f"  Parquet: {manifest['parquet']}"
                                            if manifest["parquet"] else "  (Parquet: install pandas+pyarrow)"))
    typer.echo("final acceptance: " + json.dumps(manifest["final_acceptance"]))
    for name, path in manifest["figures"].items():
        typer.echo(f"  {name}: {path}")


if __name__ == "__main__":
    app()
