"""Latency measurement — owns `LatencyRecord`, the JSONL row shape, and p50/p90/p99 rollups.

Timing rules: CUDA events for GPU stages, monotonic CPU timers for end-to-end, batch size 1.
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from policy.config import CONFIG

WARMUP_REQUESTS = CONFIG.measurement.warmup_requests            # ~50 warm-ups (discarded)
MIN_MEASURED_REQUESTS = CONFIG.measurement.min_measured_requests  # = the unique replay set (50)
PERCENTILES = CONFIG.measurement.percentiles                   # p50, p90, p99 summaries


@dataclass
class LatencyRecord:
    """One measured request. Field names match the JSONL `latency_ms` block."""
    request_id: int
    task: str
    episode_id: int
    # --- GPU/CPU stage timings (ms) ---
    preprocess_ms: float = 0.0
    h2d_ms: float = 0.0
    reasoner_ms: float = 0.0
    generator_prepare_ms: float = 0.0
    denoising_ms: float = 0.0
    denoising_step_ms: list = field(default_factory=list)
    postprocess_ms: float = 0.0
    d2h_ms: float = 0.0
    server_ms: float = 0.0            # server-side complete inference
    transport_ms: float = 0.0         # client/server communication
    first_action_ms: float = 0.0      # observation ready -> first action submission
    total_chunk_ms: float = 0.0       # observation ready -> complete 32-action chunk
    peak_memory_mb: float = 0.0
    output_checksum: str = ""
    quality_gate: str = "n/a"         # passed | failed | n/a

    def stage_breakdown(self) -> dict:
        """The six stage-breakdown buckets (preprocess folds h2d; postprocess folds d2h)."""
        return {
            "preprocess": self.preprocess_ms + self.h2d_ms,
            "reasoner_conditioning": self.reasoner_ms,
            "generator_prepare": self.generator_prepare_ms,
            "action_denoising": self.denoising_ms,
            "action_postprocess": self.postprocess_ms + self.d2h_ms,
            "transport": self.transport_ms,
        }

    def to_jsonl_row(self, *, run_id: str, configuration: str, engine: str) -> dict:
        """The minimal log format — one row per request."""
        return {
            "run_id": run_id,
            "configuration": configuration,
            "engine": engine,
            "task": self.task,
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "latency_ms": {
                "preprocess": round(self.preprocess_ms, 3),
                "h2d": round(self.h2d_ms, 3),
                "reasoner": round(self.reasoner_ms, 3),
                "generator_prepare": round(self.generator_prepare_ms, 3),
                "denoising": round(self.denoising_ms, 3),
                "postprocess": round(self.postprocess_ms, 3),
                "d2h": round(self.d2h_ms, 3),
                "transport": round(self.transport_ms, 3),
                "server_total": round(self.server_ms, 3),
                "first_action": round(self.first_action_ms, 3),
                "chunk_total": round(self.total_chunk_ms, 3),
            },
            "denoising_step_ms": [round(s, 3) for s in self.denoising_step_ms],
            "peak_memory_mb": round(self.peak_memory_mb, 1),
            "output_checksum": self.output_checksum,
            "quality_gate": self.quality_gate,
        }

    def as_dict(self) -> dict:
        return asdict(self)


# The scalar latency fields we summarize with percentiles (denoising_step_ms is an array).
SUMMARY_FIELDS = (
    "preprocess_ms", "h2d_ms", "reasoner_ms", "generator_prepare_ms", "denoising_ms",
    "postprocess_ms", "d2h_ms", "server_ms", "transport_ms",
    "first_action_ms", "total_chunk_ms", "peak_memory_mb",
)


def percentiles(values, pcts=PERCENTILES) -> dict:
    if not values:
        return {f"p{p}": 0.0 for p in pcts}
    arr = np.asarray(values, dtype=float)
    return {f"p{p}": round(float(np.percentile(arr, p)), 3) for p in pcts}


def summarize(records: list[LatencyRecord]) -> dict:
    """p50/p90/p99 per latency field over all measured requests."""
    out = {"n_requests": len(records)}
    for f in SUMMARY_FIELDS:
        out[f] = percentiles([getattr(r, f) for r in records])
    # per-step denoise timing (flatten every step of every request)
    steps = [s for r in records for s in r.denoising_step_ms]
    out["denoising_step_ms"] = percentiles(steps)
    # quality-gate tally
    gates = [r.quality_gate for r in records]
    out["quality_gate"] = {g: gates.count(g) for g in set(gates)}
    return out


# Timing primitives — torch imported lazily so the module imports with no torch/CUDA.
@contextlib.contextmanager
def cpu_timer(sink: dict, key: str):
    """Monotonic CPU timer (end-to-end stages). Writes elapsed ms into sink[key]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = (time.perf_counter() - start) * 1e3


class CudaStageTimer:
    """CUDA-event stage timer (real backend). Single synchronize, so it does not serialize
    the stream per stage."""

    def __init__(self):
        import torch  # local import: GPU-only path
        self._torch = torch
        self._events: dict = {}
        self._order: list = []

    def mark(self, name: str):
        ev = self._torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events[name] = ev
        self._order.append(name)

    def elapsed_ms(self) -> dict:
        self._torch.cuda.synchronize()
        out = {}
        for a, b in zip(self._order, self._order[1:]):
            out[f"{a}->{b}"] = self._events[a].elapsed_time(self._events[b])
        return out
