#!/usr/bin/env python
"""Run ONE optimization configuration (specification_revised.txt §4 Job 1 subprocess).

Invoked once per configuration by run_matrix.py, in its own process so the CUDA context is
released on exit. Loads the model, warms up, replays the fixed latency set, and writes the
five §7 artifacts (<cid>.jsonl, summary.json, environment.json, system-info.json,
status.json) into <out-dir>/<out-subdir>/.

    python run_configuration.py --configuration P0 --backend mock --out-dir results
"""
from __future__ import annotations

from pathlib import Path

import typer

from policy.configs import config_by_id
from policy.dataset import load_manifest, tile_to
from policy.runner import run_configuration

app = typer.Typer(add_completion=False)


@app.command()
def main(
    configuration: str = typer.Option(..., "--configuration", help="config id, e.g. P0 / E5 / E6"),
    backend: str = typer.Option("mock", "--backend", help="mock | pytorch | vllm"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir"),
    out_subdir: str = typer.Option(None, "--out-subdir", help="override output subdir (default: cid)"),
    run_id: str = typer.Option("cosmos-droid-001", "--run-id"),
    model: str = typer.Option("nvidia/Cosmos3-Nano-Policy-DROID", "--model"),
    endpoint: str = typer.Option(None, "--endpoint", help="vllm: deployed policy endpoint URL"),
    checkpoint_dir: str = typer.Option("/local/model", "--checkpoint-dir", help="pytorch: local model checkpoint"),
    manifest: Path = typer.Option(Path("policy/mock/manifest.json"), "--manifest"),
    replay_size: int = typer.Option(50, "--replay-size", help="measured requests (= unique obs; no cycling)"),
    warmups: int = typer.Option(50, "--warmups"),
    torchinductor_root: str = typer.Option("/tmp/torchinductor", "--torchinductor-root"),
    is_baseline: bool = typer.Option(False, "--is-baseline", help="tag as a §8 drift baseline"),
) -> None:
    config = config_by_id(configuration)
    # Load the replay manifest; measure replay_size requests (default = the unique set, once
    # each). A small --replay-size (e.g. --smoke's 1) takes the first N; a larger one cycles.
    requests = tile_to(load_manifest(manifest), replay_size)
    info = run_configuration(
        config, requests, backend=backend, out_dir=out_dir, run_id=run_id, model=model,
        endpoint=endpoint, checkpoint_dir=checkpoint_dir, warmups=warmups, is_baseline=is_baseline,
        torchinductor_root=torchinductor_root, out_subdir=out_subdir,
    )
    typer.echo(f"[{configuration}] measured {info['measured']} requests -> {info['dir']}")


if __name__ == "__main__":
    app()
