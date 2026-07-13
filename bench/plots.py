"""Plotting — the contribution waterfall + stage breakdown (specification.md §4).

Style follows gpu_and_inference_hw/hw3/engine_utils.py (Agg backend, shared palette,
savefig dpi=150). The waterfall is the headline deliverable: one panel per operating
point, each a descending staircase of cumulative latency with the marginal % drop of
every technique annotated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from bench.ablation import AblationResult

# Professional palette (brand reference — sequential blue + teal).
COLOR_BAR = "#1a80bb"      # lossless — medium blue
COLOR_LOSSY = "#8cc5e3"    # lossy (quality-guarded) — light blue
COLOR_SCALING = "#298c8c"  # scaling / multi-GPU (e.g. CFG-Parallel) — teal
COLOR_ANNOT = "black"      # contribution (%) labels
COLOR_GRID = "#d9d9d9"
COLOR_STOCK = "#555555"    # "stock vLLM" reference line (end of the default-on prefix)


def _stock_index(rows_data: list[dict]) -> int:
    """Index of the last variant in the contiguous run of vLLM default-on techniques
    (i.e. the cumulative 'stock vLLM' config). 0 if there is no such prefix."""
    k = 0
    for i in range(1, len(rows_data)):
        if not rows_data[i].get("default_on"):
            break
        k = i
    return k


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

    drew_stock = False
    for idx, op in enumerate(ops):
        ax = axes[idx // cols][idx % cols]
        rows_data = result.marginal_rows(op.name)
        labels = [r["variant"] for r in rows_data]
        lat = [r["ms"] for r in rows_data]
        colors = [_bar_color(r) for r in rows_data]

        x = range(len(labels))
        ax.set_axisbelow(True)                                   # grid behind the bars
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
        ax.bar(x, lat, color=colors, zorder=3)                   # opaque bars over the grid

        # annotate EVERY technique's marginal % drop vs the previous variant
        for i in range(1, len(lat)):
            drop = 100.0 * (1.0 - lat[i] / lat[i - 1]) if lat[i - 1] > 0 else 0.0
            txt = f"-{drop:.0f}%" if drop >= 0.5 else (f"+{-drop:.0f}%" if drop <= -0.5 else "0%")
            ax.annotate(txt, (i, lat[i]), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color=COLOR_ANNOT)

        # "stock vLLM" line: cumulative latency after the default-on features. Bars at/left
        # of it are what vLLM ships enabled; bars to the right are opt-in latency wins (FP8).
        k = _stock_index(rows_data)
        if k:
            drew_stock = True
            ax.axhline(lat[k], color=COLOR_STOCK, linestyle="--", linewidth=1.0, zorder=4)
            ax.annotate("stock vLLM", (len(labels) - 1, lat[k]), textcoords="offset points",
                        xytext=(0, 3), ha="right", va="bottom", fontsize=7,
                        color=COLOR_STOCK, fontstyle="italic")

        ax.set_title(op.label(), fontsize=11)
        ax.set_ylabel("latency (ms)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    # hide unused axes
    for j in range(len(ops), rows * cols):
        axes[j // cols][j % cols].axis("off")

    # single overall legend — only the categories actually present
    all_rows = [r for op in ops for r in result.marginal_rows(op.name)]
    legend_items = [(COLOR_BAR, "lossless")]
    if any(r["lossy"] and not r["scaling"] for r in all_rows):
        legend_items.append((COLOR_LOSSY, "lossy (quality-guarded)"))
    if any(r["scaling"] for r in all_rows):
        legend_items.append((COLOR_SCALING, "scaling (multi-GPU)"))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in legend_items]
    labels_l = [lbl for _, lbl in legend_items]
    if drew_stock:                                    # dashed reference line -> Line2D handle
        handles.append(Line2D([0], [0], color=COLOR_STOCK, linestyle="--"))
        labels_l.append("stock vLLM (defaults on)")
    fig.legend(handles, labels_l,
               loc="lower center", ncol=len(handles), fontsize=10, frameon=False)

    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_stage_breakdown(stage_times_ms: dict[str, float], out_path: str | Path,
                         title: str = "Stage breakdown") -> Path:
    """Stacked single-bar of where wall-clock goes (fed by bench.stages).

    Stub for Part 1 step 5 — real per-stage timings come from bench.stages.StageTimer.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    bottom = 0.0
    for name, ms in stage_times_ms.items():
        ax.bar("end-to-end", ms, bottom=bottom, label=name)
        bottom += ms
    ax.set_ylabel("latency (ms)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
