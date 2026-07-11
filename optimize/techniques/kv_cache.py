"""KV cache (Reasoner R2) — real-backend wiring stub.

Eager path: pass `use_cache=True` and switch to 1-token incremental forwards.
vLLM: on by default (paged). This hook documents the eager-path change so the
readable ablation (R0->R2) reproduces the win end-to-end.
"""
from __future__ import annotations


def apply(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg.setdefault("eager", {})["use_cache"] = True
    return cfg
