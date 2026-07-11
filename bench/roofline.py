"""Roofline classification of dominant stages (specification.md §4, F6).

Extends gpu_and_inference_hw/hw1 GPU_SPECS with H200/H100 tensor-core (BF16/FP8)
peaks + HBM bandwidth, and classifies a stage compute- vs memory-bound from its
arithmetic intensity (FLOP/byte) relative to the ridge point.
"""
from __future__ import annotations

from dataclasses import dataclass

# Approximate dense tensor-core peaks (per GPU). Verify against your SKU before quoting.
GPU_SPECS = {
    "H200": {
        "label": "NVIDIA H200 141GB HBM3e",
        "peak_flops_bf16": 989e12,   # ~0.99 PFLOP/s BF16 (dense)
        "peak_flops_fp8": 1979e12,   # ~1.98 PFLOP/s FP8 (dense)
        "peak_bw": 4.8e12,           # 4.8 TB/s HBM3e
    },
    "H100": {
        "label": "NVIDIA H100 80GB HBM3 (SXM)",
        "peak_flops_bf16": 989e12,
        "peak_flops_fp8": 1979e12,
        "peak_bw": 3.35e12,          # 3.35 TB/s HBM3
    },
}


@dataclass(frozen=True)
class Classification:
    stage: str
    arithmetic_intensity: float   # FLOP / byte
    ridge_point: float            # FLOP / byte at the roof knee
    bound: str                    # "compute" | "memory"
    headroom_note: str


def ridge_point(gpu: str, precision: str = "bf16") -> float:
    spec = GPU_SPECS[gpu]
    peak = spec["peak_flops_fp8"] if precision == "fp8" else spec["peak_flops_bf16"]
    return peak / spec["peak_bw"]


def classify(stage: str, arithmetic_intensity: float, gpu: str = "H200",
             precision: str = "bf16") -> Classification:
    rp = ridge_point(gpu, precision)
    bound = "compute" if arithmetic_intensity >= rp else "memory"
    note = ("above ridge -> compute-bound; fusion/quant helps"
            if bound == "compute"
            else "below ridge -> memory-bound; batching/weight-reuse (e.g. CFG B=2) helps")
    return Classification(stage, arithmetic_intensity, rp, bound, note)
