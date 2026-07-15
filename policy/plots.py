"""Figures (required outputs).

  plot_waterfall          Native PyTorch (P0-P3) / end-to-end (E0-E4)
                          contribution waterfalls — descending staircase, marginal %
                          drop per rung, 95% bootstrap CI whiskers.
  plot_stage_breakdown    Baseline vs final stacked stage composition (the six stages).
  plot_quality_comparison Lossy FP8 rung — speedup + gate verdict + drift.

Style mirrors the repo's prior figures (Agg backend, sequential blue palette, dpi=150).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_BAR = "#1a80bb"       # lossless
COLOR_LOSSY = "#8cc5e3"     # lossy (quality-guarded)
COLOR_FAIL = "#c8102e"      # gate failed
COLOR_ANNOT = "black"
COLOR_GRID = "#d9d9d9"
_STAGE_COLORS = {
    "preprocess": "#b8b8b8",
    "reasoner_conditioning": "#1a80bb",
    "generator_prepare": "#7bafd4",
    "action_denoising": "#298c8c",
    "action_postprocess": "#8cc5e3",
    "transport": "#e39a1a",
}


def _save(fig, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_waterfall(data: dict, out_path: str | Path, title: str | None = None) -> Path:
    rungs = data["rungs"]
    labels = [f"{r['cid']}\n{r['added'] or 'baseline'}" for r in rungs]
    p50 = [r["p50_ms"] for r in rungs]
    colors = [COLOR_LOSSY if r["lossy"] else COLOR_BAR for r in rungs]
    # asymmetric CI whiskers
    err_lo = [max(0.0, r["p50_ms"] - r["ci95_low"]) for r in rungs]
    err_hi = [max(0.0, r["ci95_high"] - r["p50_ms"]) for r in rungs]

    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(rungs) + 2), 5.5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    x = range(len(rungs))
    ax.bar(x, p50, color=colors, zorder=3,
           yerr=[err_lo, err_hi], capsize=3, ecolor="#444444")

    for i in range(1, len(p50)):
        drop = 100.0 * (1.0 - p50[i] / p50[i - 1]) if p50[i - 1] > 0 else 0.0
        txt = f"-{drop:.0f}%" if drop >= 0.5 else (f"+{-drop:.0f}%" if drop <= -0.5 else "0%")
        ax.annotate(txt, (i, p50[i]), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=COLOR_ANNOT, fontweight="bold")
    # cumulative speedup vs V0 on the final bar
    if rungs and rungs[-1]["vs_v0"]:
        ax.annotate(f"{rungs[-1]['vs_v0']:.1f}× vs baseline",
                    (len(rungs) - 1, p50[-1]), textcoords="offset points", xytext=(0, 24),
                    ha="center", fontsize=9, color=COLOR_BAR, fontweight="bold")

    # derive the rung range from the actual data (ladder length is not fixed)
    span = f"{rungs[0]['cid']}→{rungs[-1]['cid']}" if rungs else ""
    name = {"native": f"Native PyTorch ({span})",
            "end_to_end": f"End-to-end cumulative ({span})"}.get(data["waterfall"], data["waterfall"])
    ax.set_title(title or f"Cosmos3-Nano-Policy-DROID — {name} latency waterfall", fontsize=12)
    ax.set_ylabel(data["metric_label"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_BAR),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_LOSSY)]
    ax.legend(handles, ["lossless", "lossy (quality-gated)"], fontsize=9, frameon=False)
    plt.tight_layout()
    return _save(fig, out_path)


def plot_stage_breakdown(data: dict, out_path: str | Path, title: str | None = None) -> Path:
    """Baseline vs final: stacked composition across the six stages."""
    from policy.configs import STAGES
    cols = [("baseline", data["baseline_cid"], data["baseline"]),
            ("final", data["final_cid"], data["final"])]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    xs = range(len(cols))
    for xi, (_, cid, stages) in enumerate(cols):
        bottom = 0.0
        for st in STAGES:
            val = stages.get(st, 0.0)
            ax.bar(xi, val, bottom=bottom, color=_STAGE_COLORS[st], zorder=3,
                   label=st.replace("_", " ") if xi == 0 else None, width=0.55)
            if val > 0.02 * (data["baseline_total_ms"] or 1):
                ax.text(xi, bottom + val / 2, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="white")
            bottom += val
        ax.text(xi, bottom, f"  {bottom:.0f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{lbl}\n({cid})" for lbl, cid, _ in cols])
    ax.set_ylabel("latency (ms)")
    speedup = (data["baseline_total_ms"] / data["final_total_ms"]
               if data.get("final_total_ms") else 0.0)
    ax.set_title(title or f"Stage breakdown — baseline vs final ({speedup:.1f}× end-to-end)",
                 fontsize=12)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    plt.tight_layout()
    return _save(fig, out_path)


def plot_quality_comparison(data: dict, out_path: str | Path, title: str | None = None) -> Path:
    rows = data["rows"]
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(rows) + 2), 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    xs = range(len(rows))
    speedups = [r["speedup_vs_baseline"] or 0.0 for r in rows]
    colors = [COLOR_BAR if r["quality_gate"] == "passed" else COLOR_FAIL for r in rows]
    ax.bar(xs, speedups, color=colors, zorder=3, width=0.6)
    for i, r in enumerate(rows):
        ax.annotate(f"{r['quality_gate']}\ndrift={r['action_drift']}",
                    (i, speedups[i]), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color=COLOR_ANNOT)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{r['cid']}\n{r['label'].replace('+ ', '')}" for r in rows], fontsize=8)
    ax.set_ylabel("chunk-latency speedup vs baseline (×)")
    ax.set_title(title or "Lossy-technique quality gate (FP8)", fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_BAR),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_FAIL)]
    ax.legend(handles, ["gate passed", "gate failed"], fontsize=9, frameon=False)
    plt.tight_layout()
    return _save(fig, out_path)
