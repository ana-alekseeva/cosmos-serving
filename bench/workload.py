"""Operating-point matrix — fixed, representative shapes (specification.md §5).

Serving latency is shape-driven, not content-driven, so we fix
`(input_len, output_len, concurrency, resolution)` and cover the regimes the report
measures, so each technique is seen where it acts.

Two towers, two very different experiments (mirroring the report):
  - Generator (§5.3.1/§5.3.3): a per-clip *latency* waterfall over T2I/T2V/I2V x
    resolution (the tower the report quantifies), plus a separate batching
    *throughput* sweep reproducing Table 9. `baseline_latency_ms` anchors the mock.
  - Reasoner (§5.3.2): no technique ladder — a stock-vLLM concurrency/shape sweep
    (TTFT + throughput vs concurrency), mirroring `inference_benchmarks.md`. The mock
    derives TTFT/throughput analytically from the shape (see bench.drivers), so the
    reasoner OPs only carry the fixed shape x concurrency.
"""
from __future__ import annotations

from dataclasses import dataclass

from optimize.registry import GENERATOR, REASONER


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    tower: str
    description: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    modality: str                 # "text" | "image" | "video"
    baseline_latency_ms: float    # naive-baseline per-clip latency (generator mock anchor; unused for reasoner)
    clip_frames: int = 0
    clip_resolution: str = ""
    shared_prefix_tokens: int = 0  # >0 -> a shared system prompt across requests (reasoner fleet regime)
    batch_max: int = 1             # generator: max admissible batch under the 74k-token context (Table 9)

    def label(self) -> str:
        return f"{self.name} — {self.description}"


# ---------------------------------------------------------------------------
# Generator (Part 1b) — the report's quantified serving story.
# Latency OPs: task x resolution. Different techniques dominate at each point —
# T2I is host-launch-bound (CUDA graphs shine); high-res/long video is denoise- and
# decode-bound (Cache-DiT, FP8, VAE-patch, CFG-Parallel). I2V ~= action-conditioned
# world-model rollout (the robotics regime). Baselines are naive-PyTorch per-clip ms.
# ---------------------------------------------------------------------------
GENERATOR_OPS: list[OperatingPoint] = [
    OperatingPoint("t2i-1024", GENERATOR, "T2I 1024px (image)",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="image", baseline_latency_ms=3_000.0,
                   clip_frames=1, clip_resolution="1024px", batch_max=8),
    OperatingPoint("t2v-256", GENERATOR, "T2V 256p, 189 frames",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="video", baseline_latency_ms=10_000.0,
                   clip_frames=189, clip_resolution="256p", batch_max=6),
    OperatingPoint("t2v-480", GENERATOR, "T2V 480p, 189 frames",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="video", baseline_latency_ms=42_000.0,
                   clip_frames=189, clip_resolution="480p", batch_max=3),
    OperatingPoint("i2v-480", GENERATOR, "I2V 480p (action rollout)",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="video", baseline_latency_ms=84_000.0,
                   clip_frames=189, clip_resolution="480p", batch_max=3),
    OperatingPoint("t2v-720", GENERATOR, "T2V 720p, 189 frames (headline)",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="video", baseline_latency_ms=240_000.0,
                   clip_frames=189, clip_resolution="720p", batch_max=1),
]

# Report Table 9 — throughput gain (%) from request batching on the T2V task (189
# frames), by (model / hardware). The batching sweep reproduces this directly (mock),
# or measures B=1 vs B=batch_max throughput on the real backend. 720p is omitted: the
# 74k-token context admits only B=1, so batching yields no gain there.
BATCHING_TABLE9: dict[str, dict[str, int]] = {
    "t2v-256": {"Nano/H100": 8, "Nano/GB200": 40, "Super/H100": 55, "Super/GB200": 9},
    "t2v-480": {"Nano/H100": 2, "Nano/GB200": 2,  "Super/H100": 5,  "Super/GB200": 1},
}

# ---------------------------------------------------------------------------
# Reasoner (Part 1a) — stock-vLLM concurrency/shape sweep (§5.3.2). Not a technique
# ablation: the report inherits Qwen3-VL serving from vLLM out of the box, so we simply
# characterize TTFT + throughput across concurrency at fixed shapes (à la
# inference_benchmarks.md). Two shapes x concurrency {1, 64, 128, 256}.
# ---------------------------------------------------------------------------
REASONER_CONCURRENCIES: tuple[int, ...] = (1, 64, 128, 256)

# (shape_key, description, input_tokens, output_tokens, modality, clip_frames, resolution)
_REASONER_SHAPES: list[tuple] = [
    ("txt", "text in/out (in=512,out=128)", 512, 128, "text", 0, ""),
    ("vid", "video clip in / short out (in=4096,out=64)", 4096, 64, "video", 16, "256p"),
]


def _build_reasoner_ops() -> list[OperatingPoint]:
    ops: list[OperatingPoint] = []
    for key, desc, in_tok, out_tok, modality, frames, res in _REASONER_SHAPES:
        for conc in REASONER_CONCURRENCIES:
            ops.append(OperatingPoint(
                f"{key}-c{conc}", REASONER, f"{desc} @ concurrency {conc}",
                input_tokens=in_tok, output_tokens=out_tok, concurrency=conc,
                modality=modality, baseline_latency_ms=0.0,
                clip_frames=frames, clip_resolution=res,
                # a fleet at high concurrency shares a common system preamble -> prefix caching acts
                shared_prefix_tokens=1024 if conc > 1 else 0))
    return ops


REASONER_OPS: list[OperatingPoint] = _build_reasoner_ops()

_OPS: dict[str, list[OperatingPoint]] = {
    REASONER: REASONER_OPS,
    GENERATOR: GENERATOR_OPS,
}


def ops_for(tower: str) -> list[OperatingPoint]:
    if tower not in _OPS:
        raise ValueError(f"unknown tower {tower!r}")
    return _OPS[tower]


def op_by_name(tower: str, name: str) -> OperatingPoint:
    for op in ops_for(tower):
        if op.name == name:
            return op
    raise ValueError(f"unknown operating point {name!r} for {tower}")


def reasoner_shape_keys() -> list[str]:
    """Distinct reasoner shapes (e.g. ['txt', 'vid']) — one sweep curve per shape."""
    return [s[0] for s in _REASONER_SHAPES]


def batching_ops() -> list[OperatingPoint]:
    """Generator OPs that admit batching (batch_max > 1) — the batching-sweep inputs."""
    return [op for op in GENERATOR_OPS if op.batch_max > 1]


def synthetic_request(op: OperatingPoint, seed: int = 0) -> dict:
    """Deterministic request spec for `op` (metadata for mock; real driver builds tensors)."""
    return {
        "op": op.name,
        "tower": op.tower,
        "seed": seed,
        "input_tokens": op.input_tokens,
        "output_tokens": op.output_tokens,
        "concurrency": op.concurrency,
        "modality": op.modality,
        "clip_frames": op.clip_frames,
        "clip_resolution": op.clip_resolution,
    }
