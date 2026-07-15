#!/usr/bin/env python
"""Report figures from the matrix results dir and profiler traces.

    python analyze_traces.py --results-dir results --traces results/traces
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

# Categorical palette; contrast WARNs on slots 2/3 relieved by direct % labels + CSV.
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
# Ordered blue ramp for the ordinal rungs E0..E4.
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#256abf", "#184f95", "#0d366b"]
GRID = "#d9d9d9"
E_ORDER = ["E0", "E1", "E2", "E3", "E4"]

# model-part attribution (leaf-first, first match wins). Under torch.compile the python
# stack collapses at the graph boundary, so component splits are meaningful only on EAGER
# rungs (E0/E1); compiled rungs land in "compiled region" and kernel_composition covers them.
MODEL_PARTS: list[tuple[str, re.Pattern]] = [
    # GEMMs deliberately have NO leaf bucket: an aten::mm leaf stays unmatched so the owning
    # module/transformer frame above it claims the projection — that's what makes the split meaningful.
    ("attention (SDPA/FA3 + QKV)", re.compile(
        r"scaled_dot_product_attention|attention/backends|attention/layer\.py|fa3|flash|fmha"
        r"|causalattention|crossattention", re.I)),
    ("RoPE", re.compile(r"_rotate_half|_apply_rotary_pos_emb|rotary", re.I)),
    ("norms/modulation", re.compile(r"norm\.py|rms_norm|rmsnorm|layer_norm|modulat", re.I)),
    ("MLP", re.compile(r"silu|gatedmlp|gated_mlp|activation", re.I)),
    ("VAE", re.compile(r"autoencoder|(^|[^a-z])vae", re.I)),
    ("vision encoder", re.compile(r"qwen3_vl|vision|visual", re.I)),
    ("guardrail", re.compile(r"guardrail", re.I)),
    # NB: no pybind11 pattern — FA3's leaf is ALSO a pybind fwd; leave it unmatchable so the frame above decides.
    ("compiled region (fused transformer)", re.compile(
        r"_dynamo|_inductor|eval_frame|output_code|cudagraph|cuda_graph|graphs\.py", re.I)),
    ("scheduler/sampling", re.compile(r"unipc|scheduler|denoise_step|sample_actions", re.I)),
    ("transformer (other)", re.compile(r"transformer_cosmos3|pipeline_cosmos3|embed|conv", re.I)),
]

# transformer_cosmos3.py class line ranges (vllm-omni 0.24.0): frames name only
# "file.py(LINE): forward", so class ownership of the projection GEMMs comes from the line number.
_TF_LINE = re.compile(r"transformer_cosmos3\.py\((\d+)\)")
_TF_LINE_MAP: list[tuple[int, int, str]] = [
    (44, 53, "norms/modulation"),                    # RMSNorm
    (152, 195, "transformer (other)"),               # DomainAwareLinear
    (309, 385, "RoPE"),                              # rotary embedding + helpers
    (386, 416, "transformer (other)"),               # TimestepEmbedder
    (417, 460, "MLP"),                               # Cosmos3GatedMLP
    (461, 735, "attention (SDPA/FA3 + QKV)"),        # Causal + Cross attention
    (736, 954, "transformer (other)"),               # decoder layers / language model glue
    (955, 10_000, "transformer (other)"),            # VFMTransformer top level
]


def _frame_part(frame: str) -> str | None:
    m = _TF_LINE.search(frame)
    if m:
        line = int(m.group(1))
        for lo, hi, part in _TF_LINE_MAP:
            if lo <= line <= hi:
                return part
    for part, pat in MODEL_PARTS:
        if pat.search(frame):
            return part
    return None

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
            hit = _frame_part(frame)
            if hit:
                break
        if hit is None:
            unmatched[frames[-1] if frames else "?"] += usec
            hit = "other"
        parts[hit] += usec
    top_unmatched = sorted(unmatched.items(), key=lambda kv: -kv[1])[:8]
    return dict(parts), top_unmatched


def _load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        return json.load(f).get("traceEvents", [])


def _kernel_shares(events: list[dict]) -> dict[str, float]:
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


def _model_parts_from_trace(events: list[dict]) -> tuple[dict[str, float], list[tuple[str, float]]]:
    """Attribute each GPU kernel's time to the python frame stack active at its LAUNCH.

    kernel.args.correlation -> cuda_runtime launch event (CPU tid, ts) -> deepest enclosing
    python_function frame matching a MODEL_PARTS pattern (timestamp sweep per thread).
    Fallback for jobs whose stacks_cuda_rank*.txt export came back empty."""
    launches: dict[int, tuple[int, float]] = {}       # correlation -> (tid, ts)
    py_by_tid: dict[int, list[tuple[float, float, str]]] = defaultdict(list)
    kernels: list[tuple[float, str, int]] = []        # (dur, name, correlation)
    for ev in events:
        cat = str(ev.get("cat", ""))
        # cuBLAS launches its kernels via cuda_driver (cuLaunchKernelEx), NOT cuda_runtime —
        # indexing only cuda_runtime silently drops every GEMM (50-84% of kernel time).
        if cat in ("cuda_runtime", "cuda_driver"):
            corr = (ev.get("args") or {}).get("correlation")
            if corr is not None:
                launches[corr] = (ev.get("tid", 0), float(ev.get("ts", 0.0)))
        elif cat == "python_function" and ev.get("ph") == "X":
            py_by_tid[ev.get("tid", 0)].append(
                (float(ev.get("ts", 0.0)), float(ev.get("dur", 0.0)), str(ev.get("name", ""))))
        elif "kernel" in cat.lower() and ev.get("ph") == "X":
            corr = (ev.get("args") or {}).get("correlation")
            if corr is not None:
                kernels.append((float(ev.get("dur", 0.0)), str(ev.get("name", "")), corr))

    # per-tid sweep: replay frames + queries in ts order, keep the active frame stack
    queries_by_tid: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for ki, (dur, name, corr) in enumerate(kernels):
        if corr in launches:
            tid, ts = launches[corr]
            queries_by_tid[tid].append((ts, ki))
    parts: dict[str, float] = defaultdict(float)
    unmatched: dict[str, float] = defaultdict(float)
    for tid, queries in queries_by_tid.items():
        frames = sorted(py_by_tid.get(tid, []))
        queries.sort()
        stack: list[tuple[float, str]] = []           # (end_ts, name), outermost..innermost
        fi = 0
        for ts, ki in queries:
            while fi < len(frames) and frames[fi][0] <= ts:
                f_ts, f_dur, f_name = frames[fi]
                if f_ts + f_dur >= ts:                # still active at query time
                    while stack and stack[-1][0] < f_ts:
                        stack.pop()
                    stack.append((f_ts + f_dur, f_name))
                fi += 1
            while stack and stack[-1][0] < ts:
                stack.pop()
            hit = None
            for _, f_name in reversed(stack):         # deepest matching frame wins
                hit = _frame_part(f_name)
                if hit:
                    break
            dur = kernels[ki][0]
            if hit is None:
                leaf = stack[-1][1] if stack else "<no python frame>"
                unmatched[leaf] += dur
                hit = "other"
            parts[hit] += dur
    top = sorted(unmatched.items(), key=lambda kv: -kv[1])[:10]
    return dict(parts), top


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

    # --- denoise-step comparison: eager baseline vs final (each of the 4 steps faster) --
    step_cfgs = [c for c in cids if any(r.get("denoising_step_ms") for r in records[c])]
    pair = [c for c in (cids[0], cids[-1]) if c in step_cfgs] or step_cfgs[:2]
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
        ax.set_title(f"Per-denoising-step latency: {pair[0]} (eager) vs {pair[1]} (final)")
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
        parts, unmatched = {}, []
        stacks = next(iter(d.glob("stacks_cuda_rank*.txt")), None)
        if stacks and stacks.stat().st_size > 0:
            parts, unmatched = _parse_stacks(stacks)
        trace = next(iter(d.glob("trace_rank*.json*")), None)
        if trace:
            events = _load_trace(trace)
            kernel_shares[cid] = _kernel_shares(events)
            if not parts:                    # stacks export was empty -> attribute from trace
                parts, unmatched = _model_parts_from_trace(events)
        if parts:
            part_shares[cid] = parts
        if show_unmatched and unmatched:
            typer.echo(f"[{cid}] top unmatched CUDA time by leaf frame:")
            for frame, usec in unmatched:
                typer.echo(f"    {usec / 1e3:10.1f} ms  {frame[:110]}")

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
