"""`cosmos optimize` — the serving experiments (specification.md §5, §7).

Workbench-shaped: mirrors npa.cli.cosmos semantics (Typer command, `--output
{text,json}`) so it can drop into `npa workbench cosmos optimize`. Two towers, two
kinds of experiment (mirroring the report):

    # Reasoner (§5.3.2): stock-vLLM concurrency/shape sweep — TTFT + throughput.
    python -m optimize.cli --tower reasoner
    python -m optimize.cli --tower reasoner --backend vllm --out-dir results

    # Generator (§5.3.1/3): per-clip latency waterfall + Table 9 batching throughput.
    python -m optimize.cli --tower generator --ablate
    # or measure one hand-picked technique subset across the latency OPs:
    python -m optimize.cli --tower generator --enable reasoner-cache,cuda-graphs,fp8,cfg-parallel
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from bench.ablation import print_summary, run_ablation
from bench.drivers import make_engine
from bench.sweep import (
    print_batching_sweep,
    print_reasoner_sweep,
    run_batching_sweep,
    run_reasoner_sweep,
)
from bench.workload import op_by_name, ops_for
from optimize.registry import GENERATOR, PRESETS, REASONER, TOWERS, resolve

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
    preset: str = typer.Option(None, "--preset", help="none | full (generator only)"),
    enable: str = typer.Option(None, "--enable", help="comma-separated technique keys (generator only)"),
    ablate: bool = typer.Option(False, "--ablate", help="generator: cumulative latency waterfall + Table 9 batching"),
    backend: str = typer.Option("mock", "--backend", help="mock | vllm"),
    op: str = typer.Option(None, "--op", help="restrict to a single operating point (generator selection mode)"),
    model: str = typer.Option(None, "--model", help="HF model id (default nvidia/Cosmos3-Nano)"),
    port: int = typer.Option(8000, "--port", help="server port (vllm backend)"),
    repeats: int = typer.Option(10, "--repeats", help="timed runs per measurement"),
    out_dir: Path = typer.Option(Path("results"), "--out-dir", help="artifact directory"),
    output: str = typer.Option("text", "--output", help="text | json"),
) -> None:
    if tower not in TOWERS:
        _fail(f"unknown tower {tower!r}; expected one of {sorted(TOWERS)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Reasoner: stock-vLLM concurrency/shape sweep (no technique ladder) ----------
    if tower == REASONER:
        if preset or enable or ablate:
            typer.secho("note: the reasoner is served by stock vLLM (§5.3.2) — it has no "
                        "technique toggles; running the concurrency sweep.", fg=typer.colors.YELLOW, err=True)
        # --op is a substring filter over OP names (e.g. "vid" = both video shapes; "o100" =
        # all output-100 points; "vid1-o100-c256" = one point). Default: the full 32-point sweep.
        sweep_ops = ops_for(REASONER)
        if op:
            sweep_ops = [o for o in sweep_ops if op in o.name]
            if not sweep_ops:
                _fail(f"--op {op!r} matched no reasoner OPs; try a substring like 'vid', 'txt', 'o100'")
        result = run_reasoner_sweep(backend=backend, ops=sweep_ops, model=model, port=port, repeats=repeats)
        from bench.plots import plot_reasoner_sweep
        png = plot_reasoner_sweep(result, out_dir / "reasoner_sweep.png")
        js = out_dir / "reasoner_sweep.json"
        js.write_text(json.dumps(result.to_dict(), indent=2))
        if output == "text":
            print_reasoner_sweep(result)
            typer.echo(f"\nsweep plot -> {png}\nsweep json -> {js}")
        else:
            _emit({**result.to_dict(), "plot": str(png), "json": str(js)}, output)
        return

    # -- Generator: latency waterfall + Table 9 batching throughput -------------------
    if ablate:
        _run_generator_ablation(backend, op, model, port, repeats, out_dir, output)
        return

    # Selection mode: measure a chosen subset / preset across the latency OPs.
    ops = [op_by_name(GENERATOR, op)] if op else ops_for(GENERATOR)
    try:
        enabled = resolve(GENERATOR, preset=preset, enable=enable.split(",") if enable else None)
    except ValueError as exc:
        _fail(str(exc))
    engine = make_engine(backend, enabled, tower=GENERATOR, model=model, port=port)
    try:
        measured = {o.name: engine.measure(o).as_dict() for o in ops}
    finally:
        engine.close()
    lossy = [t.key for t in enabled if t.lossy]
    lines = [
        f"tower=generator backend={backend} "
        f"techniques={[t.key for t in enabled] or '(none / baseline)'}",
        f"quality-guard required for: {lossy or '(none)'}",
        "",
        f"{'OP':<10}{'p50 ms':>12}{'p95 ms':>12}",
    ]
    for name, m in measured.items():
        lines.append(f"{name:<10}{m['p50_ms']:>12.1f}{m['p95_ms']:>12.1f}")
    _emit({
        "_text": "\n".join(lines),
        "tower": GENERATOR, "backend": backend,
        "techniques": [t.key for t in enabled], "lossy_guard": lossy,
        "measurements": measured,
    }, output)


def _run_generator_ablation(backend, op, model, port, repeats, out_dir, output):
    """The generator's two headline figures: latency waterfall + Table 9 batching."""
    ops = [op_by_name(GENERATOR, op)] if op else ops_for(GENERATOR)
    js = out_dir / "generator_ablation.json"
    js_full = out_dir / "generator_ablation_full.json"

    def _on_variant(res, v):
        js.write_text(json.dumps(res.to_dict(), indent=2))
        js_full.write_text(json.dumps(res.to_full_dict(), indent=2))
        cells = "  ".join(
            f"{o.name}={res.p50[(v.index, o.name)]:.0f}ms" if (v.index, o.name) in res.p50
            else f"{o.name}=FAIL" for o in res.ops)
        typer.echo(f"[generator] variant {v.index + 1}/{len(res.variants)} done — {v.label}: {cells}")

    result = run_ablation(GENERATOR, backend=backend, ops=ops, model=model, port=port,
                          repeats=repeats, on_variant=_on_variant)
    from bench.plots import plot_batching_throughput, plot_contribution_waterfall
    wf_png = plot_contribution_waterfall(result, out_dir / "generator_waterfall.png")
    js.write_text(json.dumps(result.to_dict(), indent=2))
    js_full.write_text(json.dumps(result.to_full_dict(), indent=2))

    # Table 9: batching is a throughput result (not a latency rung), measured separately.
    batching = run_batching_sweep(backend=backend, model=model, port=port)
    bt_png = plot_batching_throughput(batching, out_dir / "generator_batching.png")
    bt_js = out_dir / "generator_batching.json"
    bt_js.write_text(json.dumps(batching.to_dict(), indent=2))

    if output == "text":
        print_summary(result)
        print_batching_sweep(batching)
        if result.failed:
            typer.echo(f"\n⚠ {len(result.failed)} measurement(s) failed — check the vLLM/AIPerf log:")
            for label, _ in result.failed:
                typer.echo(f"  - {label}")
        typer.echo(f"\nwaterfall  -> {wf_png}\nablation   -> {js}\nfull trace -> {js_full}"
                   f"\nbatching   -> {bt_png}, {bt_js}")
    else:
        _emit({**result.to_dict(), "waterfall": str(wf_png), "ablation": str(js),
               "ablation_full": str(js_full), "batching": batching.to_dict(),
               "batching_plot": str(bt_png)}, output)


if __name__ == "__main__":
    app()
