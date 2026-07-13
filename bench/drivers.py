"""Measurement drivers.

- MockEngine: computes modeled latency/TTFT/throughput from fixed shapes so the whole
  harness runs with no GPU / no model (default for local development).
- VLLMEngine: real path — launch vLLM (Reasoner) / vLLM-Omni (Generator) with the
  technique flags, measure with AIPerf (reasoner) or a timed generation request
  (generator). Written against the docs; every `# VERIFY` confirmed on-box.
- time_generation(): synchronized wall-clock timer, adapted from
  gpu_and_inference_hw/hw2/utils.py.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from bench.workload import OperatingPoint
from optimize.registry import GENERATOR, REASONER, Technique

# Mock reasoner model constants (illustrative — real numbers come from AIPerf on the H200).
# Anchored to the report's ballpark: ~2,800 tok/s at high concurrency, low tens-of-ms TTFT.
_PEAK_TOK_S = 3000.0        # aggregate decode throughput ceiling
_TOK_S_HALF = 40.0          # concurrency at which throughput reaches half of peak
_PREFILL_BASE_MS = 20.0     # fixed prefill overhead
_PREFILL_PER_TOK_MS = 0.008
_VISION_ENCODE_MS = 35.0    # extra prefill for image/video inputs (ViT encode)
_TPOT_BASE_MS = 4.0         # per-output-token decode time at concurrency 1


@dataclass(frozen=True)
class Measurement:
    op: str
    p50_ms: float
    p95_ms: float
    samples_ms: tuple[float, ...] = ()   # raw per-repeat latencies (empty if the backend can't expose them)
    ttft_ms: float = 0.0                 # reasoner sweep: time-to-first-token
    throughput_tok_s: float = 0.0        # reasoner sweep: aggregate decode throughput

    def as_dict(self) -> dict:
        return asdict(self)


def _mock_reasoner(op: OperatingPoint) -> Measurement:
    """Analytic TTFT / throughput / per-request latency vs shape + concurrency.

    Continuous batching: TTFT rises modestly with concurrency (queueing); aggregate
    throughput saturates toward a peak; per-request TPOT grows as the GPU is shared."""
    conc = max(1, op.concurrency)
    log2c = math.log2(conc)
    prefill = _PREFILL_BASE_MS + _PREFILL_PER_TOK_MS * op.input_tokens
    if op.modality != "text":
        prefill += _VISION_ENCODE_MS
    ttft = prefill * (1.0 + 0.15 * log2c)
    tput = _PEAK_TOK_S * conc / (conc + _TOK_S_HALF)
    tpot = _TPOT_BASE_MS * (1.0 + 0.10 * log2c)
    p50 = ttft + op.output_tokens * tpot
    return Measurement(op.name, round(p50, 3), round(p50 * 1.15, 3),
                       samples_ms=(round(p50, 3),),
                       ttft_ms=round(ttft, 3), throughput_tok_s=round(tput, 1))


class MockEngine:
    """Models per-clip latency (generator) or TTFT/throughput (reasoner) from fixed shapes."""

    backend = "mock"

    def __init__(self, enabled: list[Technique], *, tower: str = REASONER, **_):
        self.enabled = list(enabled)
        self.tower = tower

    def prepare(self) -> None:
        pass

    def measure(self, op: OperatingPoint, repeats: int = 10, warmup: int = 1) -> Measurement:
        if op.tower == REASONER:
            return _mock_reasoner(op)          # stock vLLM; techniques don't apply to the reasoner sweep
        latency = op.baseline_latency_ms       # generator: baseline / prod(technique speedups)
        for t in self.enabled:
            latency /= t.mock_speedups.get(op.name, 1.0)
        p50 = round(latency, 3)
        samples = tuple(p50 for _ in range(max(1, repeats)))
        return Measurement(op.name, p50, round(latency * 1.05, 3), samples)

    def close(self) -> None:
        pass


class VLLMEngine:
    """Real backend: launch vLLM/vLLM-Omni with the technique set, measure per OP.

    Server config is a function of the technique set (not the OP), so the server is
    started lazily on first measure() and reused for every OP of the variant; the
    ablation/sweep runner calls close() between variants to free the GPU and relaunch.
    Reasoner -> AIPerf (TTFT + throughput); Generator -> timed generation request.
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

    def prepare(self) -> None:
        """Launch the server once, up front, so a launch failure dooms the variant fast."""
        self._ensure_server()

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
        return Measurement(op.name, round(r["p50_ms"], 3), round(r["p95_ms"], 3),
                           tuple(r.get("samples_ms", ())),
                           ttft_ms=round(r.get("ttft_ms", 0.0), 3),
                           throughput_tok_s=round(r.get("throughput_tok_s", 0.0), 1))

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None


_ENGINES = {"mock": MockEngine, "vllm": VLLMEngine}


def make_engine(backend: str, enabled: list[Technique], *, tower: str = REASONER,
                model: str | None = None, port: int = 8000):
    if backend not in _ENGINES:
        raise ValueError(f"unknown backend {backend!r}; expected mock | vllm")
    return _ENGINES[backend](enabled, tower=tower, model=model, port=port)


def time_generation(loop_fn, *, warmup: int = 1, repeats: int = 10) -> Measurement:
    """Synchronized wall-clock timing (adapted from hw2/utils.py::time_generation).

    Warm up (excludes CUDA init / compile), then median + p95 over `repeats`. Requires
    torch + CUDA at call time.
    """
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
