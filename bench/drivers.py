"""Measurement drivers.

- MockEngine: computes modeled latency from the registry so the whole harness
  runs with no GPU / no model (default for local development).
- VLLMEngine: real path, stubbed — wire up on the H200 (launch vLLM / vLLM-Omni
  with each technique's engine_flags; measure with GenAI-Perf / vLLM bench).
- time_generation(): synchronized wall-clock timer for the eager reference path,
  adapted from gpu_and_inference_hw/hw2/utils.py.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from bench.workload import OperatingPoint
from optimize.registry import Technique


@dataclass(frozen=True)
class Measurement:
    op: str
    p50_ms: float
    p95_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


class MockEngine:
    """Models latency as baseline / prod(technique speedups in this op)."""

    backend = "mock"

    def __init__(self, enabled: list[Technique]):
        self.enabled = list(enabled)

    def measure(self, op: OperatingPoint, repeats: int = 10, warmup: int = 1) -> Measurement:
        latency = op.baseline_latency_ms
        for t in self.enabled:
            latency /= t.mock_speedups.get(op.name, 1.0)
        # deterministic p95 spread so downstream code exercises both fields
        return Measurement(op.name, round(latency, 3), round(latency * 1.05, 3))


class VLLMEngine:
    """Real backend — not yet wired. Runs on the H200 in Phase 1."""

    backend = "vllm"

    def __init__(self, enabled: list[Technique]):
        self.enabled = list(enabled)

    def measure(self, op: OperatingPoint, repeats: int = 10, warmup: int = 1) -> Measurement:
        flags = [t.engine_flags for t in self.enabled]
        raise NotImplementedError(
            "Real vLLM backend not wired yet (Phase 1 on the H200). "
            "Launch vLLM/vLLM-Omni with these flags then measure via GenAI-Perf / "
            f"vLLM bench:\n  op={op.name} flags={flags}"
        )


_ENGINES = {"mock": MockEngine, "vllm": VLLMEngine}


def make_engine(backend: str, enabled: list[Technique]):
    if backend not in _ENGINES:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_ENGINES)}")
    return _ENGINES[backend](enabled)


def time_generation(loop_fn, *, warmup: int = 1, repeats: int = 10) -> Measurement:
    """Synchronized wall-clock timing for the eager reference path.

    Adapted from hw2/utils.py::time_generation — warm up (excludes CUDA init /
    compile), then median + p95 over `repeats`. Requires torch + CUDA at call time.
    """
    import statistics
    import time

    import torch  # local import: only needed on the GPU path

    for _ in range(warmup):
        loop_fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        loop_fn()
        torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1e3)

    samples_ms.sort()
    p50 = statistics.median(samples_ms)
    p95 = samples_ms[min(len(samples_ms) - 1, round(0.95 * (len(samples_ms) - 1)))]
    return Measurement("eager", round(p50, 3), round(p95, 3))
