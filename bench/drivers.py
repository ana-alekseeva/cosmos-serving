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
from optimize.registry import GENERATOR, REASONER, Technique


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

    def __init__(self, enabled: list[Technique], **_):
        self.enabled = list(enabled)

    def measure(self, op: OperatingPoint, repeats: int = 10, warmup: int = 1) -> Measurement:
        latency = op.baseline_latency_ms
        for t in self.enabled:
            latency /= t.mock_speedups.get(op.name, 1.0)
        # deterministic p95 spread so downstream code exercises both fields
        return Measurement(op.name, round(latency, 3), round(latency * 1.05, 3))

    def close(self) -> None:
        pass


class VLLMEngine:
    """Real backend: launch vLLM/vLLM-Omni with the technique set, measure per OP.

    Server config is a function of the technique set (not the OP), so the server is
    started lazily on first measure() and reused for every OP of the variant; the
    ablation runner calls close() between variants to free the GPU and relaunch with
    the next flag set. Reasoner -> AIPerf; Generator -> timed generation request.
    """

    backend = "vllm"

    def __init__(self, enabled: list[Technique], *, tower: str = REASONER,
                 model: str | None = None, port: int = 8000):
        self.enabled = list(enabled)
        self.tower = tower
        self.model = model
        self.port = port
        self._server = None

    def _ensure_server(self):
        if self._server is None:
            from bench.serving import DEFAULT_MODEL, start_server
            model = self.model or DEFAULT_MODEL
            self._server = start_server(model, self.tower, self.enabled, port=self.port)
        return self._server

    def measure(self, op: OperatingPoint, repeats: int = 10, warmup: int = 1) -> Measurement:
        server = self._ensure_server()
        model = self.model or "nvidia/Cosmos3-Nano"
        if self.tower == GENERATOR:
            from bench.aiperf import time_generation_request
            r = time_generation_request(server.base_url, model, op,
                                         repeats=max(3, repeats // 2), warmup=warmup)
        else:
            from bench.aiperf import run_aiperf
            r = run_aiperf(server.base_url, model, op, warmup=warmup, request_count=max(30, repeats))
        return Measurement(op.name, round(r["p50_ms"], 3), round(r["p95_ms"], 3))

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None


_ENGINES = {"mock": MockEngine, "vllm": VLLMEngine}


def make_engine(backend: str, enabled: list[Technique], *, tower: str = REASONER,
                model: str | None = None, port: int = 8000):
    if backend == "eager":
        from bench.eager import EagerEngine  # lazy: avoids a circular import
        return EagerEngine(enabled, tower=tower, model=model, port=port)
    if backend not in _ENGINES:
        raise ValueError(f"unknown backend {backend!r}; expected mock | vllm | eager")
    return _ENGINES[backend](enabled, tower=tower, model=model, port=port)


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
