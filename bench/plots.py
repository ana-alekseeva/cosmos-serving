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

from bench.ablation import AblationResult

# Professional palette (brand reference — sequential blue).
COLOR_BAR = "#1a80bb"    # lossless — medium blue
COLOR_LOSSY = "#8cc5e3"  # lossy (quality-guarded) — light blue
COLOR_ANNOT = "black"    # contribution (%) labels
COLOR_GRID = "#d9d9d9"


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
        colors = [COLOR_LOSSY if r["lossy"] else COLOR_BAR for r in rows_data]

        x = range(len(labels))
        ax.set_axisbelow(True)                                   # grid behind the bars
        ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
        ax.bar(x, lat, color=colors, zorder=3)                   # opaque bars over the grid

        # annotate EVERY technique's marginal % drop vs the previous variant
        for i in range(1, len(lat)):
            drop = 100.0 * (1.0 - lat[i] / lat[i - 1]) if lat[i - 1] > 0 else 0.0
            txt = f"-{drop:.0f}%" if drop >= 0.5 else "0%"
            ax.annotate(txt, (i, lat[i]), textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=7, color=COLOR_ANNOT)

        ax.set_title(op.label(), fontsize=11)
        ax.set_ylabel("latency (ms)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    # hide unused axes
    for j in range(len(ops), rows * cols):
        axes[j // cols][j % cols].axis("off")

    # single overall legend indicating the lossy operations
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_BAR),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_LOSSY)]
    fig.legend(handles, ["lossless", "lossy (quality-guarded)"],
               loc="lower center", ncol=2, fontsize=10, frameon=False)

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
