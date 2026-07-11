"""Ulysses Context-Parallel (Generator) — real-backend wiring stub.

Shards long image/video token sequences across GPU ranks with all-to-all attention
communication. A latency technique even when memory fits (paper §5.3.1). Lossless.
On 2 GPUs it is an ALTERNATIVE to CFG-Parallel (same "distributed" group) — the two
occupy the GPUs differently and are not stacked below 4 GPUs.
"""
from __future__ import annotations


def apply(cfg: dict) -> dict:
    cfg = dict(cfg)
    omni = cfg.setdefault("vllm_omni", {})
    omni["context_parallel"] = True
    omni["ulysses_degree"] = 2
    cfg["gpus"] = max(cfg.get("gpus", 1), 2)
    return cfg
