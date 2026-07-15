#!/usr/bin/env python
"""Trace + results analysis figures (report plots beyond aggregate.py's §2 set).

Consumes the matrix results dir (per-config JSONL) plus the job's profiler traces
(`traces.tar.gz` from ${OUTPUT_URI}raw/, or an extracted directory) and produces:

  model_parts.png/.csv        which parts of the MODEL are slowest — CUDA time %
                              attributed to components (vision encoder, UND language
                              tower, GEN attention/MLP, VAE, guardrail, ...) from the
                              with_stack flamegraph (stacks_cuda_rank0.txt)
  kernel_composition.png/.csv GEMM / attention / fused / elementwise / memory shares
                              per rung, from the Chrome trace kernel events
  pareto_latency_vram.png     p50 latency vs peak VRAM per rung (E3-vs-E6 optima)
  latency_ecdf.png            per-request latency ECDFs with p50/p99 markers
  denoise_steps.png           per-denoise-step latency, rung vs rung (Cache-DiT story)
  request_timeline.png        latency vs request index (drift/bias-control evidence)

    python analyze_traces.py --results-dir results --traces results/traces.tar.gz
"""
from __future__ import annotations

import gzip
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

# Validated categorical palette (dataviz six-checks, light surface; contrast WARNs on
# slots 2/3 are relieved by direct % labels + the CSV emitted next to every figure).
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
# Ordered blue ramp for the ordinal rungs E0..E6 (lines start at step 250 for contrast).
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRID = "#d9d9d9"
E_ORDER = ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]

# --- model-part attribution (leaf-first, first match wins) -------------------------
# Patterns target the verified module names in vllm_omni/diffusion/models/cosmos3 and
# the vendored cosmos_framework; refine with --show-unmatched on real traces.
MODEL_PARTS: list[tuple[str, re.Pattern]] = [
    ("guardrail", re.compile(r"guardrail", re.I)),
    ("vae", re.compile(r"autoencoder|(^|[^a-z])vae", re.I)),
    ("vision encoder", re.compile(r"vision|visual", re.I)),
    ("UND language tower", re.compile(r"language_model|und_expert|qwen.*text|embed_prefix", re.I)),
    ("GEN attention", re.compile(r"crossattention|causalattention|attention|fa3|flash|attn", re.I)),
    ("GEN MLP", re.compile(r"gatedmlp|(^|[^a-z])mlp", re.I)),
    ("GEN blocks (other)", re.compile(r"transformer_cosmos3|gendecoderlayer|gen_layers|rotary|rope|timestep|modulat", re.I)),
    ("scheduler/sampling", re.compile(r"unipc|scheduler|denoise_step|sample_actions", re.I)),
    ("action pre/post", re.compile(r"robolab|pose_utils|action_transform|postprocess", re.I)),
]

KERNEL_CATS: list[tuple[str, re.Pattern]] = [
    ("GEMM", re.compile(r"nvjet|gemm|cublas|splitk|cutlass(?!.*flash)", re.I)),
    ("attention", re.compile(r"flash|fa3|fmha|attn", re.I)),
    ("fused (triton)", re.compile(r"^triton_", re.I)),
    ("elementwise/norm", re.compile(r"elementwise|vectorized|reduce_kernel|rms_norm|norm|gemv", re.I)),
    ("memory/copy", re.compile(r"memcpy|memset|catarray|copy|to_copy", re.I)),
]


def _records(results_dir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for cid in E_ORDER:
        f = results_dir / cid / f"{cid}.jsonl"
        if f.exists():
            out[cid] = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    return out


def _trace_dirs(traces: Path, workdir: Path) -> dict[str, Path]:
    """cid -> newest per-config profiler dir (accepts traces.tar.gz or a directory)."""
    root = traces
    if traces.is_file():  # tarball from run_job.sh
        root = workdir / "traces_extracted"
        if not root.exists():
            root.mkdir(parents=True)
            with tarfile.open(traces) as tf:
                tf.extractall(root, filter="data")
    found: dict[str, Path] = {}
    for cid in E_ORDER:
        base = root / cid
        if not base.is_dir():
            continue
        stage_dirs = sorted(d for d in base.iterdir() if d.is_dir() and "diffusion" in d.name)
        if stage_dirs:
            found[cid] = stage_dirs[-1]
    return found


def _parse_stacks(path: Path) -> tuple[dict[str, float], list[tuple[str, float]]]:
    """Flamegraph lines 'frame;frame;... <usec>' -> CUDA usec per model part."""
    parts: dict[str, float] = defaultdict(float)
    unmatched: dict[str, float] = defaultdict(float)
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or " " not in line:
            continue
        stack, _, val = line.rpartition(" ")
        try:
            usec = float(val)
        except ValueError:
            continue
        frames = stack.split(";")
        hit = None
        for frame in reversed(frames):          # leaf-first: specific beats generic
            for name, pat in MODEL_PARTS:
                if pat.search(frame):
                    hit = name
                    break
            if hit:
                break
        if hit is None:
            unmatched[frames[-1] if frames else "?"] += usec
            hit = "other"
        parts[hit] += usec
    top_unmatched = sorted(unmatched.items(), key=lambda kv: -kv[1])[:8]
    return dict(parts), top_unmatched


def _parse_trace_kernels(path: Path) -> dict[str, float]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        events = json.load(f).get("traceEvents", [])
    cats: dict[str, float] = defaultdict(float)
    for ev in events:
        if ev.get("ph") != "X" or "kernel" not in str(ev.get("cat", "")).lower():
            continue
        name, dur = str(ev.get("name", "")), float(ev.get("dur", 0.0))
        for cat, pat in KERNEL_CATS:
            if pat.search(name):
                cats[cat] += dur
                break
        else:
            cats["other"] += dur
    return dict(cats)


def _save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"  {out}")


def _stacked_shares(shares: dict[str, dict[str, float]], order: list[str], title: str,
                    out_png: Path, out_csv: Path) -> None:
    """Horizontal 100% stacked bars (one per config) + CSV table (contrast relief)."""
    cids = [c for c in E_ORDER if c in shares]
    # full-detail CSV rows from the UNfolded shares (the chart may fold the tail below)
    all_cols = [c for c in order if any(shares[k].get(c) for k in cids)] + ["other"]
    csv_rows = []
    for cid in cids:
        total = sum(shares[cid].values()) or 1.0
        csv_rows.append({"configuration": cid,
                         **{c: round(100.0 * shares[cid].get(c, 0.0) / total, 2) for c in all_cols}})
    present = [c for c in order if any(shares[k].get(c) for k in cids)]
    # never cycle categorical hues: top-N components keep named slots, tail folds into
    # "other" on the CHART (the CSV keeps every component at full detail).
    if len(present) > len(CAT):
        ranked = sorted(present, key=lambda c: -sum(shares[k].get(c, 0.0) for k in cids))
        keep = set(ranked[: len(CAT)])
        for k in cids:
            folded = sum(v for c, v in shares[k].items() if c not in keep and c != "other")
            shares[k] = {c: v for c, v in shares[k].items() if c in keep or c == "other"}
            shares[k]["other"] = shares[k].get("other", 0.0) + folded
        present = [c for c in order if c in keep]
    cols = present + ["other"]
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(cids) + 2.2))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for yi, cid in enumerate(reversed(cids)):
        total = sum(shares[cid].values()) or 1.0
        left = 0.0
        for ci, col in enumerate(cols):
            pct = 100.0 * shares[cid].get(col, 0.0) / total
            color = CAT[ci] if col != "other" else "#b8b8b8"
            ax.barh(yi, pct, left=left, color=color, height=0.55,
                    edgecolor="white", linewidth=2, zorder=3)
            if pct >= 6:                      # direct labels on the big segments
                ax.text(left + pct / 2, yi, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            left += pct
    ax.set_yticks(range(len(cids)), list(reversed(cids)))
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of CUDA time (%)")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1,
                             color=(CAT[i] if c != "other" else "#b8b8b8"))
               for i, c in enumerate(cols)]
    ax.legend(handles, cols, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(5, len(cols)), fontsize=8, frameon=False)
    _save(fig, out_png)
    out_csv.write_text(json.dumps(csv_rows, indent=2))
    typer.echo(f"  {out_csv}")


def _pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    i = max(0, min(len(s) - 1, round(q * (len(s) - 1))))
    return s[int(i)]


app = typer.Typer(add_completion=False)


@app.command()
def main(
    results_dir: Path = typer.Option(Path("results"), "--results-dir"),
    traces: Path = typer.Option(None, "--traces", help="traces.tar.gz or extracted dir"),
    out_dir: Path = typer.Option(None, "--out-dir", help="default <results-dir>/aggregate"),
    show_unmatched: bool = typer.Option(False, "--show-unmatched",
                                        help="print top CUDA frames not matched to a model part"),
) -> None:
    out = out_dir or results_dir / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    records = _records(results_dir)
    if not records:
        raise typer.Exit(typer.echo(f"no per-config JSONL under {results_dir}") or 1)
    cids = [c for c in E_ORDER if c in records]
    totals = {c: [r["latency_ms"]["chunk_total"] for r in records[c]] for c in cids}

    # --- latency ECDFs (ordinal ramp; legend + selective endpoint labels) ----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.8)
    for i, cid in enumerate(cids):
        xs = sorted(totals[cid])
        ys = [(k + 1) / len(xs) for k in range(len(xs))]
        color = RAMP[i % len(RAMP)]
        ax.plot(xs, ys, color=color, linewidth=2, label=cid)
        if cid in (cids[0], min(cids, key=lambda c: _pct(totals[c], .5)), cids[-1]):
            ax.annotate(cid, (xs[-1], 1.0), textcoords="offset points", xytext=(4, -10 * (i % 3)),
                        fontsize=8, color=color, fontweight="bold")
    ax.set_xlabel("total chunk latency (ms)")
    ax.set_ylabel("fraction of requests ≤ x")
    ax.set_title("Per-request latency ECDF by configuration (p50/p99 in summary.csv)")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out / "latency_ecdf.png")

    # --- Pareto: p50 latency vs peak VRAM ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.8)
    for cid in cids:
        p50 = _pct(totals[cid], .5)
        vram = max((r.get("peak_memory_mb") or 0.0) for r in records[cid]) / 1024.0
        ax.scatter(p50, vram, s=90, color="#2a78d6", zorder=3)
        ax.annotate(cid, (p50, vram), textcoords="offset points", xytext=(7, 4),
                    fontsize=9, fontweight="bold")
    ax.set_xlabel("p50 total chunk latency (ms)")
    ax.set_ylabel("peak GPU memory (GB)")
    ax.set_title("Latency–memory Pareto by rung (lower-left is better)")
    _save(fig, out / "pareto_latency_vram.png")

    # --- request-index timeline (drift evidence) ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for i, cid in enumerate(cids):
        ax.plot(range(len(totals[cid])), totals[cid], ".", markersize=5,
                color=RAMP[i % len(RAMP)], alpha=0.8, label=cid)
    ax.set_xlabel("request index within configuration run")
    ax.set_ylabel("total chunk latency (ms)")
    ax.set_title("Per-request latency in measurement order (flat = no drift)")
    ax.legend(fontsize=8, frameon=False, ncol=len(cids))
    _save(fig, out / "request_timeline.png")

    # --- denoise-step comparison (Cache-DiT story) ----------------------------------
    step_cfgs = [c for c in cids if any(r.get("denoising_step_ms") for r in records[c])]
    pair = [c for c in ("E3", "E5") if c in step_cfgs] or step_cfgs[:2]
    if len(pair) == 2:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        width = 0.38
        for gi, cid in enumerate(pair):
            per_step = [r["denoising_step_ms"] for r in records[cid] if r.get("denoising_step_ms")]
            n = min(len(s) for s in per_step)
            med = [_pct([s[k] for s in per_step], .5) for k in range(n)]
            xs = [k + (gi - 0.5) * width for k in range(n)]
            ax.bar(xs, med, width=width, color=CAT[gi * 2], zorder=3, label=cid,
                   edgecolor="white", linewidth=2)
            for x, v in zip(xs, med):
                ax.annotate(f"{v:.0f}", (x, v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(n), [f"step {k + 1}" for k in range(n)])
        ax.set_ylabel("median step latency (ms)")
        ax.set_title(f"Denoising step latency: {pair[0]} vs {pair[1]} "
                     "(no step gets cheaper ⇒ cache never activates)")
        ax.legend(fontsize=9, frameon=False)
        _save(fig, out / "denoise_steps.png")

    # --- trace-derived figures -------------------------------------------------------
    if traces is None or not Path(traces).exists():
        typer.echo("no --traces given/found — skipped model_parts + kernel_composition")
        raise typer.Exit(0)
    tdirs = _trace_dirs(Path(traces), out)
    if not tdirs:
        typer.echo(f"no per-config profiler dirs under {traces} — expected <cid>/<ts>_stage_0_*_diffusion_*/")
        raise typer.Exit(0)

    part_shares: dict[str, dict[str, float]] = {}
    kernel_shares: dict[str, dict[str, float]] = {}
    for cid, d in tdirs.items():
        stacks = next(iter(d.glob("stacks_cuda_rank*.txt")), None)
        if stacks:
            parts, unmatched = _parse_stacks(stacks)
            part_shares[cid] = parts
            if show_unmatched and unmatched:
                typer.echo(f"[{cid}] top unmatched CUDA frames:")
                for frame, usec in unmatched:
                    typer.echo(f"    {usec / 1e3:10.1f} ms  {frame[:110]}")
        trace = next(iter(d.glob("trace_rank*.json*")), None)
        if trace:
            kernel_shares[cid] = _parse_trace_kernels(trace)

    if part_shares:
        _stacked_shares(part_shares, [n for n, _ in MODEL_PARTS],
                        "Where the model spends CUDA time (from with_stack traces)",
                        out / "model_parts.png", out / "model_parts.csv")
    if kernel_shares:
        _stacked_shares(kernel_shares, [n for n, _ in KERNEL_CATS],
                        "Kernel-time composition by rung (GEMM-bound workload)",
                        out / "kernel_composition.png", out / "kernel_composition.csv")


if __name__ == "__main__":
    app()
