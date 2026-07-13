"""`cosmos optimize` — selectable engine optimization (specification.md §5, §7).

Workbench-shaped: mirrors npa.cli.cosmos semantics (Typer command, `--output
{text,json}`) so it can drop into `npa workbench cosmos optimize`. One command
serves three jobs — pick a subset, pick the full package, or run the ablation:

    python -m optimize.cli --tower reasoner --preset full
    python -m optimize.cli --tower reasoner --enable kv-cache,cuda-graphs,fp8,evs
    python -m optimize.cli --tower reasoner --ablate --out-dir results
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from bench.ablation import print_summary, run_ablation
from bench.drivers import make_engine
from bench.workload import op_by_name, ops_for
from optimize.registry import PRESETS, TOWERS, resolve

app = typer.Typer(add_completion=False, help="Cosmos 3 serving optimization.")


def _fail(msg: str) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _emit(payload: dict, output: str) -> None:
    if output == "json":
        typer.echo(json.dumps(payload, indent=2))
    else:
        typer.echo(payload.get("_text", json.dumps(payload, indent=2)))


@app.command("optimize")
def optimize_cmd(
    tower: str = typer.Option("reasoner", "--tower", help="reasoner | generator"),
    preset: str = typer.Option(None, "--preset", help="none | full"),
    enable: str = typer.Option(None, "--enable", help="comma-separated technique keys"),
    ablate: bool = typer.Option(False, "--ablate", help="run the cumulative ladder + waterfall"),
    backend: str = typer.Option("mock", "--backend", help="mock | vllm | eager"),
    op: str = typer.Option(None, "--op", help="restrict to a single operating point"),
    model: str = typer.Option(None, "--model", help="HF model id (default nvidia/Cosmos3-Nano)"),
    port: int = typer.Option(8000, "--port", help="server port (vllm backend)"),
    repeats: int = typer.Option(10, "--repeats", help="timed runs per measurement (eager: 2-3 is plenty — it's deterministic)"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="artifact directory"),
    output: str = typer.Option("text", "--output", help="text | json"),
) -> None:
    """Select techniques (or `--preset full`) for `--tower`, or `--ablate` the ladder."""
    if tower not in TOWERS:
        _fail(f"unknown tower {tower!r}; expected one of {sorted(TOWERS)}")
    if preset and preset not in PRESETS:
        _fail(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")

    ops = [op_by_name(tower, op)] if op else ops_for(tower)

    if ablate:
        js = out_dir / f"{tower}_ablation.json"
        js_full = out_dir / f"{tower}_ablation_full.json"
        js.parent.mkdir(parents=True, exist_ok=True)

        def _on_variant(res, v):  # persist partial JSON (summary + full trace) after each variant
            js.write_text(json.dumps(res.to_dict(), indent=2))
            js_full.write_text(json.dumps(res.to_full_dict(), indent=2))
            cells = "  ".join(
                f"{o.name}={res.p50[(v.index, o.name)]:.0f}ms" if (v.index, o.name) in res.p50
                else f"{o.name}=FAIL" for o in res.ops)   # an OP can fail without dropping the variant
            typer.echo(f"[{tower}] variant {v.index + 1}/{len(res.variants)} done — {v.label}: {cells}")

        result = run_ablation(tower, backend=backend, ops=ops, model=model, port=port,
                              repeats=repeats, on_variant=_on_variant)
        from bench.plots import plot_contribution_waterfall  # lazy: needs matplotlib
        png = plot_contribution_waterfall(result, out_dir / f"{tower}_waterfall.png")
        js.write_text(json.dumps(result.to_dict(), indent=2))
        js_full.write_text(json.dumps(result.to_full_dict(), indent=2))
        if output == "text":
            print_summary(result)
            if result.failed:
                fix = {
                    "eager": "install the missing package on the GPU host (e.g. flash-attn "
                             "for FlashAttention, or the fp8 quant deps) — see bench/eager.py — and re-run",
                    "vllm": "check the AIPerf/vLLM log for the failed cell (heavy concurrency OPs can "
                            "time out AIPerf startup); raise the timeout or the flag, then re-run",
                }.get(backend, "resolve the error above and re-run")
                typer.echo(f"\n⚠ {len(result.failed)} measurement(s) failed — {fix}:")
                for label, _ in result.failed:
                    typer.echo(f"  - {label}")
            typer.echo(f"\nwaterfall  -> {png}\nablation   -> {js}\nfull trace -> {js_full}")
        else:
            _emit({**result.to_dict(), "waterfall": str(png), "ablation": str(js),
                   "ablation_full": str(js_full)}, output)
        return

    # Selection mode: measure the chosen subset / preset across the operating points.
    try:
        enabled = resolve(tower, preset=preset,
                          enable=enable.split(",") if enable else None)
    except ValueError as exc:
        _fail(str(exc))

    engine = make_engine(backend, enabled, tower=tower, model=model, port=port)
    try:
        measured = {o.name: engine.measure(o).as_dict() for o in ops}
    finally:
        engine.close()
    lossy = [t.key for t in enabled if t.lossy]
    lines = [
        f"tower={tower} backend={backend} "
        f"techniques={[t.key for t in enabled] or '(none / baseline)'}",
        f"quality-guard required for: {lossy or '(none)'}",
        "",
        f"{'OP':<6}{'p50 ms':>12}{'p95 ms':>12}",
    ]
    for name, m in measured.items():
        lines.append(f"{name:<6}{m['p50_ms']:>12.1f}{m['p95_ms']:>12.1f}")

    _emit({
        "_text": "\n".join(lines),
        "tower": tower,
        "backend": backend,
        "techniques": [t.key for t in enabled],
        "lossy_guard": lossy,
        "measurements": measured,
    }, output)


if __name__ == "__main__":
    # Single-command Typer app: options are passed directly (no subcommand name).
    # The named `optimize` command is still mounted under `npa workbench cosmos`.
    app()
