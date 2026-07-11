"""Technique registry — the single source of truth for the `optimize` command.

Holds, per tower:
  - which NVIDIA techniques exist (each an independent toggle),
  - the canonical *cumulative* order used by the ablation waterfall,
  - presets (`none` = naïve baseline, `full` = every technique),
  - a small mock latency model so the whole harness runs without a GPU.

`mock_speedups` is used ONLY by bench.drivers.MockEngine so the ablation and
plots are runnable locally. On the H200 the real backend reads `engine_flags`
and measures wall-clock instead (see specification.md §5–§6).
"""
from __future__ import annotations

from dataclasses import dataclass

REASONER = "reasoner"
GENERATOR = "generator"
TOWERS = (REASONER, GENERATOR)


@dataclass(frozen=True)
class Technique:
    key: str                 # CLI token, e.g. "kv-cache"
    label: str               # human label for tables/plots
    tower: str               # REASONER | GENERATOR
    lossy: bool              # True -> triggers the quality guard
    regime: str              # primary operating point where it acts (UX/doc)
    engine_flags: dict       # real-backend flags (wired on the GPU)
    mock_speedups: dict      # op_name -> latency divisor (>1 = faster); mock only
    note: str = ""


# ---------------------------------------------------------------------------
# Reasoner ladder (Part 1a) — canonical cumulative order.
# op names: A (latency), B (decode), C (throughput), D (multimodal / robot).
# ---------------------------------------------------------------------------
REASONER_TECHNIQUES: list[Technique] = [
    Technique("inference-mode", "inference_mode", REASONER, False, "A",
              {"eager": {"inference_mode": True}},
              {"A": 1.10, "B": 1.10, "C": 1.05, "D": 1.08, "E": 1.08, "F": 1.06}),
    Technique("kv-cache", "KV cache", REASONER, False, "B",
              {"eager": {"use_cache": True}},
              {"A": 2.50, "B": 6.00, "C": 2.00, "D": 1.50, "E": 1.50, "F": 1.60}),
    Technique("deferred-sync", "deferred sampling sync", REASONER, False, "A",
              {"eager": {"defer_sync": True}},
              {"A": 1.30, "B": 1.15, "C": 1.05, "D": 1.05, "E": 1.05, "F": 1.05}),
    Technique("cuda-graphs", "torch.compile / CUDA graphs", REASONER, False, "A",
              {"vllm": {"enforce_eager": False}},
              {"A": 1.60, "B": 1.15, "C": 1.10, "D": 1.10, "E": 1.15, "F": 1.10}),
    Technique("flash-attn", "FlashAttention / fused attention", REASONER, False, "A",
              {"vllm": {"attention_backend": "FLASH_ATTN"}},
              {"A": 1.20, "B": 1.50, "C": 1.30, "D": 1.40, "E": 1.35, "F": 1.40}),
    Technique("paged-kv", "paged KV-cache", REASONER, False, "C",
              {"vllm": {"__architectural__": "paged_attention"}},
              {"A": 1.05, "B": 1.20, "C": 1.50, "D": 1.10, "E": 1.10, "F": 1.50}),
    Technique("continuous-batching", "continuous batching", REASONER, False, "C",
              {"vllm": {"__architectural__": "continuous_batching"}},
              {"A": 1.00, "B": 1.00, "C": 2.50, "D": 1.10, "E": 1.00, "F": 2.50}),
    Technique("fp8", "FP8 / NVFP4 quantization", REASONER, True, "B",
              {"vllm": {"quantization": "fp8"}},
              {"A": 1.20, "B": 1.50, "C": 1.40, "D": 1.30, "E": 1.30, "F": 1.35}),
    Technique("evs", "EVS token pruning", REASONER, True, "D",
              {"vllm_omni": {"enable_evs": True}},
              {"A": 1.00, "B": 1.00, "C": 1.00, "D": 2.20, "E": 1.40, "F": 2.20}),
]

# ---------------------------------------------------------------------------
# Generator ladder (Part 1b) — op names: R256 / R480 / R720 (resolution).
# Baseline already runs sequential CFG; cfg-parallel is the CFG *optimization*.
# ---------------------------------------------------------------------------
GENERATOR_TECHNIQUES: list[Technique] = [
    Technique("reasoner-cache", "reasoner-tower output caching", GENERATOR, False, "R256",
              {"cosmos": {"reasoner_cache": True}},
              {"R256": 1.10, "R480": 1.08, "R720": 1.06}),
    Technique("cuda-graphs", "torch.compile / CUDA graphs", GENERATOR, False, "R256",
              {"cosmos": {"cuda_graphs": True}},
              {"R256": 1.40, "R480": 1.10, "R720": 1.05}),
    Technique("cache-dit", "Cache-DiT", GENERATOR, True, "R720",
              {"vllm_omni": {"cache_dit": True}},
              {"R256": 1.30, "R480": 1.50, "R720": 1.60}),
    Technique("fp8", "FP8 quantization", GENERATOR, True, "R480",
              {"vllm_omni": {"quantization": "fp8"}},
              {"R256": 1.50, "R480": 1.50, "R720": 1.50}),
    Technique("vae-patch", "VAE-Patch-Parallel", GENERATOR, False, "R720",
              {"vllm_omni": {"vae_patch_parallel": True}},
              {"R256": 1.05, "R480": 1.10, "R720": 1.15}),
    Technique("cfg-parallel", "CFG-Parallel (2 GPU)", GENERATOR, False, "R720",
              {"vllm_omni": {"cfg_parallel": True, "gpus": 2}},
              {"R256": 1.80, "R480": 1.80, "R720": 1.80},
              note="NVIDIA technique; uses a 2nd GPU — a scaling win, not per-GPU algorithmic."),
]

_LADDERS: dict[str, list[Technique]] = {
    REASONER: REASONER_TECHNIQUES,
    GENERATOR: GENERATOR_TECHNIQUES,
}

PRESETS = ("none", "full")


def techniques_for(tower: str) -> list[Technique]:
    if tower not in _LADDERS:
        raise ValueError(f"unknown tower {tower!r}; expected one of {TOWERS}")
    return _LADDERS[tower]


def ablation_ladder(tower: str) -> list[Technique]:
    """Canonical cumulative order for the waterfall (already ordered)."""
    return list(techniques_for(tower))


def by_key(tower: str) -> dict[str, Technique]:
    return {t.key: t for t in techniques_for(tower)}


def resolve(tower: str, *, preset: str | None = None,
            enable: list[str] | None = None) -> list[Technique]:
    """Return the enabled techniques (in ladder order) for a preset or subset."""
    ladder = techniques_for(tower)
    if enable:
        wanted = {k.strip() for k in enable if k.strip()}
        known = by_key(tower)
        unknown = wanted - known.keys()
        if unknown:
            raise ValueError(
                f"unknown technique(s) for {tower}: {sorted(unknown)}; "
                f"available: {sorted(known)}"
            )
        return [t for t in ladder if t.key in wanted]  # preserve ladder order
    if preset == "full":
        return list(ladder)
    if preset == "none" or preset is None:
        return []
    raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
