"""CFG-Parallel (Generator G7) — real-backend wiring stub.

NVIDIA's technique: run the conditional forward on GPU 0 and the unconditional on
GPU 1, combine with a single P2P exchange per denoising step. Needs 2 GPUs.
Lossless (same math, split across ranks). Attribution note: its waterfall bar
mixes a 2nd GPU (scaling) with the CFG split — label accordingly (spec §N4).
"""
from __future__ import annotations


def apply(cfg: dict) -> dict:
    cfg = dict(cfg)
    omni = cfg.setdefault("vllm_omni", {})
    omni["cfg_parallel"] = True
    cfg["gpus"] = max(cfg.get("gpus", 1), 2)
    return cfg
