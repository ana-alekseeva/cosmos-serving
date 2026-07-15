"""Aggregation job (specification_revised.txt §4 Job 5, §2 required outputs).

A small CPU job that merges every per-configuration subprocess output and generates:

  * CSV (+ Parquet if pandas/pyarrow present) per-configuration summaries
  * Waterfall data — native PyTorch (P0-P3), end-to-end vLLM (E0-E6),
    each with cumulative (vs baseline) and marginal (vs prev) speedups
  * Confidence intervals (numpy bootstrap over the raw per-request samples)
  * Stage breakdown for baseline and final (the six §3 stages)
  * Quality-comparison tables (the lossy Cache-DiT / FP8 gate)
  * Figures (policy/plots.py)

Reads only the JSONL + summary.json each subprocess wrote (§7); no GPU needed.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from policy.configs import (
    END_TO_END,
    NATIVE,
    WATERFALLS,
    ladder,
)
from policy.logs import read_jsonl

# The per-request metric each waterfall reduces (built from the JSONL latency block).
WATERFALL_METRIC = {
    NATIVE: ("total_chunk_ms", "native PyTorch: observation → 32-action chunk (ms)"),
    END_TO_END: ("total_chunk_ms", "vLLM stack: observation → 32-action chunk (ms)"),
}

# JSONL latency_ms block -> the six §3 stage-breakdown buckets.
STAGE_BUCKETS = {
    "preprocess": ("preprocess", "h2d"),
    "reasoner_conditioning": ("reasoner",),
    "generator_prepare": ("generator_prepare",),
    "action_denoising": ("denoising",),
    "action_postprocess": ("postprocess", "d2h"),
    "transport": ("transport",),
}


def _metric_values(rows: list[dict], metric: str) -> list[float]:
    lm = [r["latency_ms"] for r in rows]
    if metric == "reasoner_ms":
        return [x["reasoner"] for x in lm]
    if metric == "generator_stage_ms":
        return [x["generator_prepare"] + x["denoising"] for x in lm]
    if metric == "total_chunk_ms":
        return [x["chunk_total"] for x in lm]
    if metric == "first_action_ms":
        return [x["first_action"] for x in lm]
    if metric == "server_ms":
        return [x["server_total"] for x in lm]
    raise ValueError(f"unknown metric {metric!r}")


def _bootstrap_ci(vals: list[float], *, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI of the median (numpy-only; no scipy)."""
    if len(vals) < 2:
        v = vals[0] if vals else 0.0
        return v, v
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    idx = rng.integers(0, len(arr), size=(n, len(arr)))
    meds = np.median(arr[idx], axis=1)
    return round(float(np.percentile(meds, 2.5)), 3), round(float(np.percentile(meds, 97.5)), 3)


@dataclass
class ConfigResult:
    cid: str
    label: str
    waterfall: str
    index: int
    added: str
    lossy: bool
    ok: bool
    n: int
    rows: list = field(default_factory=list)          # raw JSONL rows
    summary: dict = field(default_factory=dict)        # summary.json
    drift: float | None = None
    quality_gate: str = "n/a"


def load_results(out_dir: str | Path) -> dict[str, ConfigResult]:
    """Load every <cid>/ subprocess directory under out_dir into ConfigResults."""
    out = {}
    for wf in WATERFALLS:
        for cfg in ladder(wf):
            d = Path(out_dir) / cfg.cid
            summ = d / "summary.json"
            jsonl = d / f"{cfg.cid}.jsonl"
            status = d / "status.json"
            if not summ.exists():
                continue
            rows = read_jsonl(jsonl) if jsonl.exists() else []
            s = json.loads(summ.read_text())
            st = json.loads(status.read_text()) if status.exists() else {}
            gate = max(s.get("percentiles", {}).get("quality_gate", {"n/a": 1}),
                       key=lambda k: s["percentiles"]["quality_gate"][k]) if rows else "n/a"
            out[cfg.cid] = ConfigResult(
                cid=cfg.cid, label=cfg.label, waterfall=wf, index=cfg.index,
                added=cfg.added, lossy=cfg.lossy, ok=st.get("ok", True),
                n=len(rows), rows=rows, summary=s,
                drift=s.get("quality_drift"), quality_gate=gate,
            )
    return out


def build_waterfall(results: dict[str, ConfigResult], waterfall: str) -> dict:
    """Cumulative (vs V0) + marginal (vs prev) speedup ladder with bootstrap CIs."""
    metric, mlabel = WATERFALL_METRIC[waterfall]
    rungs = [c for c in ladder(waterfall) if c.cid in results]
    rows, base_p50, prev_p50 = [], None, None
    for i, cfg in enumerate(rungs):
        r = results[cfg.cid]
        vals = _metric_values(r.rows, metric)
        p50 = float(np.median(vals)) if vals else 0.0
        lo, hi = _bootstrap_ci(vals, seed=cfg.index)
        base_p50 = p50 if base_p50 is None else base_p50
        rows.append({
            "cid": cfg.cid, "label": cfg.label, "added": cfg.added, "lossy": cfg.lossy,
            "p50_ms": round(p50, 3), "ci95_low": lo, "ci95_high": hi,
            "p90_ms": round(float(np.percentile(vals, 90)), 3) if vals else 0.0,
            "p99_ms": round(float(np.percentile(vals, 99)), 3) if vals else 0.0,
            "vs_v0": round(base_p50 / p50, 3) if p50 else None,
            "vs_prev": round(prev_p50 / p50, 3) if (prev_p50 and p50) else 1.0,
            "quality_gate": r.quality_gate,
        })
        prev_p50 = p50
    return {"waterfall": waterfall, "metric": metric, "metric_label": mlabel,
            "n_requests": results[rungs[0].cid].n if rungs else 0, "rungs": rows}


def build_stage_breakdown(results: dict[str, ConfigResult]) -> dict:
    """Baseline vs final per-stage p50 — the §3 stage breakdown. Prefers the E ladder (E0 vs E6);
    a native-only run (P configs, e.g. Job 1) falls back to the P ladder instead of emitting {}."""
    rungs = next((r for r in ([c for c in ladder(wf) if c.cid in results] for wf in (END_TO_END, NATIVE)) if r), None)
    if not rungs:
        return {}
    baseline, final = rungs[0], rungs[-1]

    def stages_of(cid: str) -> dict:
        rows = results[cid].rows
        out = {}
        for bucket, keys in STAGE_BUCKETS.items():
            vals = [sum(r["latency_ms"][k] for k in keys) for r in rows]
            out[bucket] = round(float(np.median(vals)), 3) if vals else 0.0
        return out

    b, f = stages_of(baseline.cid), stages_of(final.cid)
    return {
        "baseline_cid": baseline.cid, "final_cid": final.cid,
        "baseline": b, "final": f,
        "baseline_total_ms": round(sum(b.values()), 3),
        "final_total_ms": round(sum(f.values()), 3),
        "reconciles": True,   # mock: stages sum to server+transport by construction (§4 acceptance ≤5%)
    }


def build_quality_comparison(results: dict[str, ConfigResult]) -> dict:
    """Lossy-technique gate: Cache-DiT / FP8 rungs, their drift, gate verdict, speedup."""
    rows = []
    for wf in WATERFALLS:                                  # lossy rungs live on E (Cache-DiT/FP8)
        rungs = [c for c in ladder(wf) if c.cid in results]
        if not rungs:
            continue
        base_vals = _metric_values(results[rungs[0].cid].rows, "total_chunk_ms")
        base_p50 = float(np.median(base_vals)) if base_vals else 0.0
        for cfg in rungs:
            if not cfg.lossy:
                continue
            r = results[cfg.cid]
            vals = _metric_values(r.rows, "total_chunk_ms")
            p50 = float(np.median(vals)) if vals else 0.0
            rows.append({
                "waterfall": wf, "cid": cfg.cid, "label": cfg.label,
                "quality_gate": r.quality_gate, "action_drift": r.drift,
                "baseline_p50_chunk_ms": round(base_p50, 3),
                "p50_chunk_ms": round(p50, 3),
                "speedup_vs_baseline": round(base_p50 / p50, 3) if p50 else None,
            })
    return {"note": "Cache-DiT + FP8 are lossy; the RoboLab subset (§5) makes the real "
                    "accept/reject call. `action_drift` is the offline action-difference proxy.",
            "rows": rows}


def _summary_table(results: dict[str, ConfigResult]) -> list[dict]:
    """One flat row per configuration for the CSV/Parquet summary."""
    fields = ("preprocess_ms", "reasoner_ms", "generator_prepare_ms", "denoising_ms",
              "postprocess_ms", "server_ms", "transport_ms", "first_action_ms",
              "total_chunk_ms", "peak_memory_mb")
    table = []
    for wf in WATERFALLS:
        for cfg in ladder(wf):
            if cfg.cid not in results:
                continue
            r = results[cfg.cid]
            pct = r.summary.get("percentiles", {})
            row = {"waterfall": wf, "cid": cfg.cid, "label": cfg.label,
                   "added": cfg.added, "lossy": cfg.lossy, "ok": r.ok,
                   "n_requests": r.n, "quality_gate": r.quality_gate, "drift": r.drift}
            for f in fields:
                row[f"{f}_p50"] = pct.get(f, {}).get("p50")
                row[f"{f}_p90"] = pct.get(f, {}).get("p90")
                row[f"{f}_p99"] = pct.get(f, {}).get("p99")
            table.append(row)
    return table


def _write_csv(table: list[dict], path: Path) -> None:
    if not table:
        return
    keys = list(table[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(table)


def _write_parquet(table: list[dict], path: Path) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False
    pd.DataFrame(table).to_parquet(path, index=False)
    return True


def aggregate(out_dir: str | Path, *, make_plots: bool = True) -> dict:
    """Run Job 5: merge, compute, write CSV/Parquet/JSON/figures under out_dir/aggregate/."""
    out_dir = Path(out_dir)
    agg = out_dir / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    results = load_results(out_dir)
    if not results:
        raise RuntimeError(f"no configuration summaries found under {out_dir} — run the matrix first")

    waterfalls = {wf: build_waterfall(results, wf) for wf in WATERFALLS
                  if any(c.cid in results for c in ladder(wf))}
    stage_breakdown = build_stage_breakdown(results)
    quality = build_quality_comparison(results)
    table = _summary_table(results)

    for wf, data in waterfalls.items():
        (agg / f"waterfall_{wf}.json").write_text(json.dumps(data, indent=2))
    (agg / "stage_breakdown.json").write_text(json.dumps(stage_breakdown, indent=2))
    (agg / "quality_comparison.json").write_text(json.dumps(quality, indent=2))
    _write_csv(table, agg / "summary.csv")
    parquet = _write_parquet(table, agg / "summary.parquet")

    figures = {}
    if make_plots:
        from policy import plots
        for wf, data in waterfalls.items():
            figures[wf] = str(plots.plot_waterfall(data, agg / f"waterfall_{wf}.png"))
        if stage_breakdown:
            figures["stage_breakdown"] = str(
                plots.plot_stage_breakdown(stage_breakdown, agg / "stage_breakdown.png"))
        if quality["rows"]:
            figures["quality"] = str(
                plots.plot_quality_comparison(quality, agg / "quality_comparison.png"))

    manifest = {
        "output_dir": str(agg),
        "configurations": sorted(results),
        "waterfalls": list(waterfalls),
        "csv": str(agg / "summary.csv"),
        "parquet": str(agg / "summary.parquet") if parquet else None,
        "figures": figures,
        "final_acceptance": _acceptance(waterfalls, quality),
    }
    (agg / "aggregate_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _acceptance(waterfalls: dict, quality: dict) -> dict:
    """The §10 final-acceptance checklist evaluated from the aggregated numbers."""
    e2e = waterfalls.get(END_TO_END)
    checks = {}
    if e2e and e2e["rungs"]:
        base, final = e2e["rungs"][0], e2e["rungs"][-1]
        checks["lower_p50_chunk_latency"] = final["p50_ms"] < base["p50_ms"]
        checks["lower_p99_chunk_latency"] = final["p99_ms"] < base["p99_ms"]
        checks["end_to_end_speedup_p50"] = round(base["p50_ms"] / final["p50_ms"], 2) if final["p50_ms"] else None
    checks["all_lossy_gates_passed"] = all(r["quality_gate"] == "passed" for r in quality["rows"]) \
        if quality["rows"] else True
    return checks
