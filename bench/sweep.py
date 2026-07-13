"""Sweep runners — the two experiments that aren't cumulative technique ablations.

  - Reasoner concurrency/shape sweep (§5.3.2): one stock-vLLM server, measure TTFT +
    throughput across concurrency {1,64,128,256} at fixed shapes. Mirrors
    inference_benchmarks.md. There is no technique ladder for the reasoner.
  - Generator batching throughput sweep (§5.3.1, Table 9): throughput gain from request
    batching on T2V at 256p / 480p (720p admits only B=1). Mock reproduces Table 9
    directly; the real backend measures B=1 vs B=batch_max throughput.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bench.drivers import make_engine
from bench.workload import (
    BATCHING_TABLE9,
    OperatingPoint,
    op_by_name,
    ops_for,
)
from optimize.registry import GENERATOR, REASONER


# ---------------------------------------------------------------------------
# Reasoner concurrency / shape sweep (1:1 with inference_benchmarks.md)
# ---------------------------------------------------------------------------
@dataclass
class ReasonerSweepResult:
    backend: str
    shapes: list[str]                          # family+output keys, e.g. ["txt-o1", "vid1-o100", ...]
    concurrencies: list[int]
    points: dict = field(default_factory=dict)  # op_name -> {shape, family, output_tokens, modality, concurrency, ttft_ms, throughput_tok_s, req_throughput_req_s, p50_ms, p95_ms}

    def curve(self, shape: str, metric: str) -> list[tuple[int, float]]:
        """(concurrency, metric) pairs for one shape key, ordered by concurrency."""
        pts = [p for p in self.points.values() if p["shape"] == shape]
        return [(p["concurrency"], p[metric]) for p in sorted(pts, key=lambda p: p["concurrency"])]

    def curve_fo(self, family: str, output_tokens: int, metric: str) -> list[tuple[int, float]]:
        """(concurrency, metric) for one modality family at one output length."""
        pts = [p for p in self.points.values()
               if p["family"] == family and p["output_tokens"] == output_tokens]
        return [(p["concurrency"], p[metric]) for p in sorted(pts, key=lambda p: p["concurrency"])]

    def families(self) -> list[str]:
        return list(dict.fromkeys(p["family"] for p in self.points.values()))

    def output_lengths(self) -> list[int]:
        return sorted({p["output_tokens"] for p in self.points.values()})

    def to_dict(self) -> dict:
        return {
            "experiment": "reasoner_concurrency_sweep",
            "backend": self.backend,
            "note": "1:1 with NVIDIA/cosmos inference_benchmarks.md: input=50, output {1,100}, "
                    "concurrency {1,64,128,256}. NVIDIA benchmarks video only; text/image added here.",
            "shapes": self.shapes,
            "concurrencies": self.concurrencies,
            "points": self.points,
        }


def _shape_of(op_name: str) -> str:
    return op_name.split("-c")[0]          # "vid1-o100-c64" -> "vid1-o100"


def _family_of(shape: str) -> str:
    return shape.split("-o")[0]            # "vid1-o100" -> "vid1"


def run_reasoner_sweep(*, backend: str = "mock", ops: list[OperatingPoint] | None = None,
                       model: str | None = None, port: int = 8000, repeats: int = 30,
                       on_point=None) -> ReasonerSweepResult:
    ops = ops or ops_for(REASONER)
    concs = sorted({op.concurrency for op in ops})
    shapes = list(dict.fromkeys(_shape_of(op.name) for op in ops))
    result = ReasonerSweepResult(backend=backend, shapes=shapes, concurrencies=concs)
    engine = make_engine(backend, [], tower=REASONER, model=model, port=port)  # stock vLLM: no techniques
    if backend != "mock":
        print(f"» reasoner sweep: stock vLLM, {len(ops)} points "
              f"({len(shapes)} shapes x concurrency {concs})", flush=True)
    engine.prepare()
    try:
        for op in ops:
            m = engine.measure(op, repeats=repeats)
            shape = _shape_of(op.name)
            result.points[op.name] = {
                "shape": shape, "family": _family_of(shape), "output_tokens": op.output_tokens,
                "modality": op.modality, "concurrency": op.concurrency,
                "ttft_ms": m.ttft_ms, "throughput_tok_s": m.throughput_tok_s,
                "req_throughput_req_s": m.req_throughput_req_s,
                "p50_ms": m.p50_ms, "p95_ms": m.p95_ms,
            }
            if backend != "mock":
                print(f"    {op.name}: TTFT={m.ttft_ms:.0f}ms  lat={m.p50_ms:.0f}ms  "
                      f"tput={m.throughput_tok_s:.0f} tok/s  {m.req_throughput_req_s:.1f} req/s", flush=True)
            if on_point is not None:
                on_point(result, op)
    finally:
        engine.close()
    return result


# ---------------------------------------------------------------------------
# Generator batching throughput sweep (Table 9)
# ---------------------------------------------------------------------------
@dataclass
class BatchingSweepResult:
    backend: str
    rows: list = field(default_factory=list)   # {resolution, batch_max, series, gain_pct}

    def to_dict(self) -> dict:
        return {
            "experiment": "generator_batching_throughput",
            "backend": self.backend,
            "note": "Throughput gain (%) from request batching on T2V, 189 frames. "
                    "720p omitted: the 74k-token context admits only B=1.",
            "rows": self.rows,
        }


def run_batching_sweep(*, backend: str = "mock", model: str | None = None,
                       port: int = 8000, waves: int = 2) -> BatchingSweepResult:
    """Reproduce Table 9. Mock: emit the report's per-(model/hw) gains directly.
    Real: measure B=1 vs B=batch_max throughput on vLLM-Omni and report the gain."""
    result = BatchingSweepResult(backend=backend)
    if backend == "mock":
        for res, gains in BATCHING_TABLE9.items():
            batch_max = op_by_name(GENERATOR, res).batch_max
            for series, pct in gains.items():
                result.rows.append({"resolution": res, "batch_max": batch_max,
                                    "series": series, "gain_pct": float(pct)})
        return result

    # Real backend: one vLLM-Omni server (stock generator config), measure both batch sizes.
    from bench.aiperf import measure_generation_throughput
    engine = make_engine(backend, [], tower=GENERATOR, model=model, port=port)
    engine.prepare()
    try:
        server = engine._ensure_server()  # reuse the one server for both batch sizes
        mdl = model or "nvidia/Cosmos3-Nano"
        for res in BATCHING_TABLE9:
            op = op_by_name(GENERATOR, res)
            t1 = measure_generation_throughput(server.base_url, mdl, op, batch=1, waves=waves)
            tb = measure_generation_throughput(server.base_url, mdl, op, batch=op.batch_max, waves=waves)
            gain = (tb / t1 - 1.0) * 100.0 if t1 > 0 else 0.0
            print(f"    {res}: B=1 {t1:.3f} clips/s -> B={op.batch_max} {tb:.3f} clips/s "
                  f"(+{gain:.0f}%)", flush=True)
            result.rows.append({"resolution": res, "batch_max": op.batch_max,
                                "series": "measured", "gain_pct": round(gain, 1),
                                "tput_b1": round(t1, 4), "tput_bmax": round(tb, 4)})
    finally:
        engine.close()
    return result


def print_reasoner_sweep(result: ReasonerSweepResult) -> None:
    """One table per shape (mirrors inference_benchmarks.md: TTFT, request latency,
    token throughput, request throughput vs concurrency)."""
    for shape in result.shapes:
        print("\n" + "=" * 66)
        print(f"[reasoner] shape={shape}   (backend={result.backend})")
        print("=" * 66)
        print(f"{'conc':>6}{'TTFT ms':>12}{'latency ms':>12}{'tok/s':>12}{'req/s':>12}")
        print("-" * 54)
        for conc, ttft in result.curve(shape, "ttft_ms"):
            pt = result.points[f"{shape}-c{conc}"]
            print(f"{conc:>6}{ttft:>12.1f}{pt['p50_ms']:>12.1f}"
                  f"{pt['throughput_tok_s']:>12.0f}{pt['req_throughput_req_s']:>12.1f}")


def print_batching_sweep(result: BatchingSweepResult) -> None:
    print("\n" + "=" * 60)
    print(f"[generator] batching throughput gain — Table 9   (backend={result.backend})")
    print("=" * 60)
    print(f"{'resolution':>12}{'B_max':>7}{'series':>14}{'gain %':>9}")
    print("-" * 42)
    for r in result.rows:
        print(f"{r['resolution']:>12}{r['batch_max']:>7}{r['series']:>14}{r['gain_pct']:>8.0f}%")
