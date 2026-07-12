"""Isolation probe: is FlashAttention's eager-path regression FA2 itself, or the
FA2 x torch.compile / CUDA-graph interaction?

The cumulative ablation only ever measures FA2 with compile ALREADY on
(variant 4 = kv-cache + cuda-graphs + flash-attn), so it cannot separate the two.
This runs the four corners on the eager backend and reports the FA2 effect twice —
once with compile off, once with compile on:

    kv                    decode baseline (KV cache, SDPA, no compile)
    kv+flash              FA2 alone            -> FA2 effect, compile OFF
    kv+compile            compile alone
    kv+compile+flash      FA2 + compile (= ablation variant 4) -> FA2 effect, compile ON

Reading it: each "FA2 effect" is a speedup (>1.0 faster, <1.0 slower) vs the same
config without flash. If compile-OFF is ~1.0 but compile-ON is <<1.0, the eager
regression is the FA2 x torch.compile interaction (graph breaks), not the kernel.
If BOTH are <<1.0, FA2 itself is slow at these single-request shapes.

    python -m bench.fa2_probe [--op A] [--repeats 10] [--warmup 2] [--out-dir results-eager]

Eager backend only (attn impl + torch.compile are baked at model load, so each
config gets a fresh load); single-request OPs A/B/D/E.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

from bench.drivers import make_engine
from bench.workload import op_by_name, ops_for
from optimize.registry import REASONER, resolve

app = typer.Typer(add_completion=False)

# The four corners. Each pair (with/without "flash-attn") isolates the FA2 effect at
# a fixed compile setting; keys are resolved to eager toggles by their key alone.
CONFIGS: list[tuple[str, list[str]]] = [
    ("kv", ["kv-cache"]),
    ("kv+flash", ["kv-cache", "flash-attn"]),
    ("kv+compile", ["kv-cache", "cuda-graphs"]),
    ("kv+compile+flash", ["kv-cache", "cuda-graphs", "flash-attn"]),
]


@app.command()
def main(
    op: str = typer.Option(None, "--op", help="single OP (default: all single-request A/B/D/E)"),
    repeats: int = typer.Option(10, "--repeats", help="timed runs per measurement"),
    warmup: int = typer.Option(2, "--warmup", help="warmup runs (>=2 lets torch.compile capture its graph)"),
    model: str = typer.Option(None, "--model", help="HF model id (default nvidia/Cosmos3-Nano)"),
    out_dir: Path = typer.Option(Path("results-eager"), "--out-dir", help="artifact directory"),
) -> None:
    """Measure the four corners and report the FA2 effect with compile off vs on."""
    ops = [op_by_name(REASONER, op)] if op else [o for o in ops_for(REASONER) if o.concurrency == 1]

    # (config_label, op_name) -> Measurement. Fresh engine per config: attn impl and
    # torch.compile are set at load time, so each config needs its own model load.
    meas: dict[tuple[str, str], object] = {}
    for label, keys in CONFIGS:
        enabled = resolve(REASONER, enable=keys)
        engine = make_engine("eager", enabled, tower=REASONER, model=model)
        try:
            for o in ops:
                meas[(label, o.name)] = engine.measure(o, repeats=repeats, warmup=warmup)
        finally:
            engine.close()
        cells = "  ".join(f"{o.name}={meas[(label, o.name)].p50_ms:.0f}ms" for o in ops)
        typer.echo(f"[{label}] {cells}")

    def p50(label: str, op_name: str) -> float:
        return meas[(label, op_name)].p50_ms

    # -- report table ------------------------------------------------------------
    labels = [c[0] for c in CONFIGS]
    hdr = f"\n{'OP':<5}" + "".join(f"{lab:>20}" for lab in labels) + f"{'FA2 off':>11}{'FA2 on':>11}"
    typer.echo(hdr)
    typer.echo("-" * len(hdr.strip("\n")))
    fa2_effect: dict[str, dict[str, float]] = {}
    for o in ops:
        row = f"{o.name:<5}" + "".join(f"{p50(lab, o.name):>20.1f}" for lab in labels)
        off = p50("kv", o.name) / p50("kv+flash", o.name)                    # FA2 effect, compile OFF
        on = p50("kv+compile", o.name) / p50("kv+compile+flash", o.name)     # FA2 effect, compile ON
        fa2_effect[o.name] = {"compile_off": round(off, 3), "compile_on": round(on, 3)}
        row += f"{off:>10.2f}x{on:>10.2f}x"
        typer.echo(row)

    typer.echo("\nFA2 off/on = FlashAttention speedup with compile OFF / ON (>1.0 faster, <1.0 slower).")
    typer.echo("  off ~1.0 but on <<1.0  -> regression is the FA2 x torch.compile/CUDA-graph interaction.")
    typer.echo("  both <<1.0             -> FA2 itself is slow at these single-request shapes.")
    typer.echo(f"(warmup={warmup}, repeats={repeats})")

    # -- persist -----------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fa2_probe.json"
    out.write_text(json.dumps({
        "backend": "eager",
        "tower": REASONER,
        "warmup": warmup,
        "repeats": repeats,
        "configs": labels,
        "ops": [o.name for o in ops],
        "measurements": {
            lab: {o.name: meas[(lab, o.name)].as_dict() for o in ops} for lab in labels
        },
        "fa2_effect": fa2_effect,
    }, indent=2))
    typer.echo(f"\nprobe -> {out}")


if __name__ == "__main__":
    app()
