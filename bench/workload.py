"""Operating-point matrix — fixed, representative shapes (specification.md §5).

Serving latency is shape-driven, not content-driven, so we fix
`(input_len, output_len, concurrency, #multimodal_tokens)` and cover the four
regimes so each technique is measured where it actually acts. `baseline_latency_ms`
anchors the mock model (the naïve R0 / G0 latency); the real backend ignores it.
"""
from __future__ import annotations

from dataclasses import dataclass

from optimize.registry import REASONER, GENERATOR


@dataclass(frozen=True)
class OperatingPoint:
    name: str
    tower: str
    description: str
    input_tokens: int
    output_tokens: int
    concurrency: int
    modality: str                 # "text" | "video"
    baseline_latency_ms: float    # naïve-baseline latency (mock anchor)
    clip_frames: int = 0
    clip_resolution: str = ""

    def label(self) -> str:
        return f"{self.name} — {self.description}"


# Reasoner: A latency / B decode / C throughput / D multimodal (robot).
# OP-D is a short video clip (stresses ViT encode + EVS pruning hardest).
REASONER_OPS: list[OperatingPoint] = [
    OperatingPoint("A", REASONER, "short text in / out",
                   input_tokens=50, output_tokens=32, concurrency=1,
                   modality="text", baseline_latency_ms=1200.0),
    OperatingPoint("B", REASONER, "short text in / long text out",
                   input_tokens=50, output_tokens=512, concurrency=1,
                   modality="text", baseline_latency_ms=9000.0),
    OperatingPoint("C", REASONER, "short text · high concurrency",
                   input_tokens=50, output_tokens=100, concurrency=128,
                   modality="text", baseline_latency_ms=6000.0),
    OperatingPoint("D", REASONER, "video clip in / short text out",
                   input_tokens=4096, output_tokens=48, concurrency=1,
                   modality="video", baseline_latency_ms=4000.0,
                   clip_frames=16, clip_resolution="256p"),
    OperatingPoint("E", REASONER, "image in / short text out",
                   input_tokens=1024, output_tokens=48, concurrency=1,
                   modality="image", baseline_latency_ms=1800.0,
                   clip_frames=1, clip_resolution="512p"),
    OperatingPoint("F", REASONER, "video clip in · high concurrency (fleet)",
                   input_tokens=4096, output_tokens=48, concurrency=128,
                   modality="video", baseline_latency_ms=9000.0,
                   clip_frames=16, clip_resolution="256p"),
]

# Generator: task x resolution points (Part 1b). Different techniques dominate:
# T2I is launch-bound (CUDA graphs); high-res video is denoise/decode-bound
# (Cache-DiT, FP8, VAE-patch, CFG-Parallel). I2V ~= action-conditioned rollout (robot).
GENERATOR_OPS: list[OperatingPoint] = [
    OperatingPoint("t2i-1024", GENERATOR, "T2I 1024px (image)",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="image", baseline_latency_ms=3_000.0,
                   clip_frames=1, clip_resolution="1024px"),
    OperatingPoint("t2v-256", GENERATOR, "T2V 256p, 189 frames",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="video", baseline_latency_ms=10_000.0,
                   clip_frames=189, clip_resolution="256p"),
    OperatingPoint("i2v-480", GENERATOR, "I2V 480p (action rollout)",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="video", baseline_latency_ms=84_000.0,
                   clip_frames=189, clip_resolution="480p"),
    OperatingPoint("t2v-720", GENERATOR, "T2V 720p, 189 frames",
                   input_tokens=0, output_tokens=0, concurrency=1,
                   modality="video", baseline_latency_ms=240_000.0,
                   clip_frames=189, clip_resolution="720p"),
]

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
