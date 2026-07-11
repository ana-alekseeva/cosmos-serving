"""One module per technique toggle (real-backend wiring).

The registry (`optimize.registry`) holds each technique's *metadata* and mock
model. These modules hold the *real* wiring: how a toggle maps onto vLLM /
vLLM-Omni / cosmos-framework launch args or code paths, applied on the H200.

Each module exposes `apply(cfg: dict) -> dict`, taking and returning the engine
launch config so techniques compose (the ablation applies them in ladder order).
Stubs below establish the pattern; fill them in during Phase 1 on the GPU.
"""
from __future__ import annotations


def apply_all(cfg: dict, technique_keys: list[str]) -> dict:
    """Fold each enabled technique's `apply` over the engine config, in order."""
    from importlib import import_module

    for key in technique_keys:
        mod_name = key.replace("-", "_")
        try:
            mod = import_module(f"optimize.techniques.{mod_name}")
        except ModuleNotFoundError:
            continue  # not all toggles need a code hook (some are pure flags)
        cfg = mod.apply(cfg)
    return cfg
