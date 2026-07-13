"""Plotting — the three headline figures (specification.md §4).

Style follows gpu_and_inference_hw/hw3/engine_utils.py (Agg backend, shared palette,
savefig dpi=150):
  - plot_contribution_waterfall : Generator per-clip latency waterfall, one panel per
    operating point, each a descending staircase with the marginal % drop annotated.
  - plot_reasoner_sweep         : Reasoner TTFT + throughput vs concurrency (§5.3.2).
  - plot_batching_throughput    : Generator batching gain vs resolution (Table 9).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.ablation import AblationResult
from bench.sweep import BatchingSweepResult, ReasonerSweepResult

# Professional palette (brand reference — sequential blue + teal).
COLOR_BAR = "#1a80bb"      # lossless — medium blue
COLOR_LOSSY = "#8cc5e3"    # lossy (quality-guarded) — light blue
COLOR_SCALING = "#298c8c"  # scaling / multi-GPU (CFG-/Context-Parallel) — teal
COLOR_ANNOT = "black"
COLOR_GRID = "#d9d9d9"
_SERIES_COLORS = ["#1a80bb", "#8cc5e3", "#298c8c", "#b8b8b8", "#e39a1a"]


def _bar_color(row: dict) -> str:
    if row.get("scaling"):
        return COLOR_SCALING
    if row.get("lossy"):
        return COLOR_LOSSY
    return COLOR_BAR


def _grid(n: int):
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), squeeze=False)
    return fig, axes, rows, cols


def plot_contribution_waterfall(result: AblationResult, out_path: str | Path,
                                title: str | None = None) -> Path:
    ops = result.ops
    fig, axes, rows, cols = _grid(len(ops))
    fig.suptitle(title or f"Cosmos 3 {result.tower.title()} — technique contribution "
                          f"(backend={result.backend})", fontsize=14, y=0.99)

    for idx, op in enumerate(ops):
        ax = axes[idx // cols][idx % cols]
        rows_data = result.marginal_rows(op.name)
        labels = [r["variant"] for r in rows_data]
        lat = [r["ms"] for r in rows_data]
        colors = [_bar_color(r) for r in rows_data]

        x = range(len(labels))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
        ax.bar(x, lat, color=colors, zorder=3)

        for i in range(1, len(lat)):                 # marginal % drop vs the previous variant
            drop = 100.0 * (1.0 - lat[i] / lat[i - 1]) if lat[i - 1] > 0 else 0.0
            txt = f"-{drop:.0f}%" if drop >= 0.5 else (f"+{-drop:.0f}%" if drop <= -0.5 else "0%")
            ax.annotate(txt, (i, lat[i]), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color=COLOR_ANNOT)

        ax.set_title(op.label(), fontsize=11)
        ax.set_ylabel("latency (ms)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    for j in range(len(ops), rows * cols):
        axes[j // cols][j % cols].axis("off")

    all_rows = [r for op in ops for r in result.marginal_rows(op.name)]
    legend_items = [(COLOR_BAR, "lossless")]
    if any(r["lossy"] and not r["scaling"] for r in all_rows):
        legend_items.append((COLOR_LOSSY, "lossy (quality-guarded)"))
    if any(r["scaling"] for r in all_rows):
        legend_items.append((COLOR_SCALING, "scaling (multi-GPU)"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in legend_items]
    fig.legend(handles, [lbl for _, lbl in legend_items],
               loc="lower center", ncol=len(handles), fontsize=10, frameon=False)

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, out_path)


def plot_reasoner_sweep(result: ReasonerSweepResult, out_path: str | Path,
                        title: str | None = None) -> Path:
    """Faceted like inference_benchmarks.md: one row per output length, columns
    {TTFT, throughput}, one curve per modality family. The throughput column shows
    request/s for the out=1 (captioning) row and token/s for the out=100 (VQA) row."""
    outputs = result.output_lengths()
    families = result.families()
    fig, axes = plt.subplots(len(outputs), 2, figsize=(14, 5 * len(outputs)), squeeze=False)
    fig.suptitle(title or f"Cosmos 3 Reasoner — stock vLLM concurrency sweep, input=50 "
                          f"(backend={result.backend})", fontsize=14, y=0.99)

    for ri, out in enumerate(outputs):
        tput_metric, tput_label = (("req_throughput_req_s", "throughput (req/s)") if out == 1
                                   else ("throughput_tok_s", "throughput (tok/s)"))
        panels = (("ttft_ms", "TTFT (ms)", f"out={out}: time-to-first-token"),
                  (tput_metric, tput_label, f"out={out}: {tput_label.split('(')[0].strip()}"))
        for ci, (metric, ylabel, title_) in enumerate(panels):
            ax = axes[ri][ci]
            ax.set_axisbelow(True)
            ax.grid(color=COLOR_GRID, linewidth=0.8)
            for fi, fam in enumerate(families):
                pts = result.curve_fo(fam, out, metric)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", color=_SERIES_COLORS[fi % len(_SERIES_COLORS)], label=fam)
            ax.set_xscale("log", base=2)
            ax.set_xticks(result.concurrencies)
            ax.set_xticklabels([str(c) for c in result.concurrencies])
            ax.set_xlabel("concurrency")
            ax.set_ylabel(ylabel)
            ax.set_title(title_, fontsize=11)
            ax.legend(title="input", fontsize=8)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out_path)


def plot_batching_throughput(result: BatchingSweepResult, out_path: str | Path,
                             title: str | None = None) -> Path:
    """Grouped bars: batching throughput gain (%) per resolution, per series (Table 9)."""
    resolutions = sorted({r["resolution"] for r in result.rows})
    series = list(dict.fromkeys(r["series"] for r in result.rows))  # preserve first-seen order
    fig, ax = plt.subplots(figsize=(max(7.5, 2.4 * len(resolutions) + 3), 5))
    ax.set_title(title or f"Cosmos 3 Generator — batching throughput gain, T2V 189f "
                          f"(Table 9, backend={result.backend})", fontsize=11)

    width = 0.8 / max(1, len(series))
    for si, s in enumerate(series):
        xs, ys = [], []
        for ri, res in enumerate(resolutions):
            match = next((r for r in result.rows if r["resolution"] == res and r["series"] == s), None)
            xs.append(ri + si * width)
            ys.append(match["gain_pct"] if match else 0.0)
        ax.bar(xs, ys, width=width, color=_SERIES_COLORS[si % len(_SERIES_COLORS)],
               label=s, zorder=3)
        for xv, yv in zip(xs, ys):
            ax.annotate(f"{yv:.0f}%", (xv, yv), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color=COLOR_ANNOT)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    ax.set_xticks([ri + width * (len(series) - 1) / 2 for ri in range(len(resolutions))])
    batch_max = {r["resolution"]: r["batch_max"] for r in result.rows}
    ax.set_xticklabels([f"{res}\n(B≤{batch_max[res]})" for res in resolutions])
    ax.set_ylabel("throughput gain vs B=1 (%)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out_path)


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
