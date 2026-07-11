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
    scaling: bool = False      # True -> uses extra GPUs (e.g. CFG-Parallel): a scaling win, not per-GPU
    category: str = "latency"  # latency | scaling | throughput | memory
    group: str = ""            # non-empty -> mutually-exclusive alternatives (one per group on N GPUs)


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
              {"A": 1.00, "B": 1.00, "C": 2.50, "D": 1.10, "E": 1.00, "F": 2.50},
              category="throughput"),
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
    Technique("reasoner-cache", "reasoner-tower output caching", GENERATOR, False, "t2i-1024",
              {"cosmos": {"reasoner_cache": True}},
              {"t2i-1024": 1.15, "t2v-256": 1.10, "i2v-480": 1.08, "t2v-720": 1.06}),
    Technique("cuda-graphs", "torch.compile / CUDA graphs", GENERATOR, False, "t2i-1024",
              {"cosmos": {"cuda_graphs": True}},
              {"t2i-1024": 1.55, "t2v-256": 1.25, "i2v-480": 1.10, "t2v-720": 1.05}),
    Technique("cache-dit", "Cache-DiT", GENERATOR, True, "t2v-720",
              {"vllm_omni": {"cache_dit": True}},
              {"t2i-1024": 1.20, "t2v-256": 1.35, "i2v-480": 1.50, "t2v-720": 1.60}),
    Technique("fp8", "FP8 quantization", GENERATOR, True, "t2v-720",
              {"vllm_omni": {"quantization": "fp8"}},
              {"t2i-1024": 1.40, "t2v-256": 1.45, "i2v-480": 1.50, "t2v-720": 1.55}),
    Technique("vae-patch", "VAE-Patch-Parallel", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"vae_patch_parallel": True}},
              {"t2i-1024": 1.05, "t2v-256": 1.08, "i2v-480": 1.12, "t2v-720": 1.18}),
    Technique("batching", "request batching (seq-packing)", GENERATOR, False, "t2v-256",
              {"cosmos": {"batching": True}},
              {"t2i-1024": 1.00, "t2v-256": 1.00, "i2v-480": 1.00, "t2v-720": 1.00},
              note="Throughput only; no benefit at B=1 (latency-bound / robotics).",
              category="throughput"),
    Technique("cfg-parallel", "CFG-Parallel (2 GPU)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"cfg_parallel": True, "gpus": 2}},
              {"t2i-1024": 1.70, "t2v-256": 1.80, "i2v-480": 1.82, "t2v-720": 1.85},
              note="NVIDIA technique; uses a 2nd GPU — a scaling win, not per-GPU algorithmic.",
              scaling=True, category="scaling", group="distributed"),
    Technique("context-parallel", "Ulysses Context-Parallel (2 GPU)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"context_parallel": True, "ulysses_degree": 2, "gpus": 2}},
              {"t2i-1024": 1.55, "t2v-256": 1.65, "i2v-480": 1.68, "t2v-720": 1.70},
              note="Alternative 2-GPU strategy to CFG-Parallel (same group; not stacked on 2 GPUs).",
              scaling=True, category="scaling", group="distributed"),
    Technique("hsdp", "HSDP (FSDP2 weight sharding)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"hsdp": True}},
              {"t2i-1024": 1.00, "t2v-256": 1.00, "i2v-480": 1.00, "t2v-720": 1.00},
              note="Memory reduction (peak VRAM); neutral on single-request latency.",
              category="memory"),
    Technique("cpu-offload", "CPU offload (layer-wise)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"enable_layerwise_offload": True}},
              {"t2i-1024": 0.80, "t2v-256": 0.80, "i2v-480": 0.80, "t2v-720": 0.80},
              note="Memory reduction; ADDS latency (PCIe transfers). Only for memory-constrained GPUs.",
              category="memory"),
]

_LADDERS: dict[str, list[Technique]] = {
    REASONER: REASONER_TECHNIQUES,
    GENERATOR: GENERATOR_TECHNIQUES,
}

PRESETS = ("none", "full")


def techniques_for(tower: str) -> list[Technique]:
    """Every toggle for the tower (for --enable / listing / --preset full)."""
    if tower not in _LADDERS:
        raise ValueError(f"unknown tower {tower!r}; expected one of {TOWERS}")
    return _LADDERS[tower]


def by_key(tower: str) -> dict[str, Technique]:
    return {t.key: t for t in techniques_for(tower)}


def _collapse_groups(techs: list[Technique]) -> list[Technique]:
    """Keep only the first technique of each mutually-exclusive group."""
    seen, out = set(), []
    for t in techs:
        if t.group:
            if t.group in seen:
                continue
            seen.add(t.group)
        out.append(t)
    return out


def full_stack(tower: str) -> list[Technique]:
    """`--preset full`: every feature, one strategy per exclusive group."""
    return _collapse_groups(techniques_for(tower))


def ablation_ladder(tower: str) -> list[Technique]:
    """Cumulative waterfall order: the full stack minus memory-only techniques
    (they don't reduce single-request latency; CPU offload adds it)."""
    return [t for t in full_stack(tower) if t.category != "memory"]


def resolve(tower: str, *, preset: str | None = None,
            enable: list[str] | None = None) -> list[Technique]:
    """Enabled techniques (in ladder order) for a preset or explicit subset."""
    all_techs = techniques_for(tower)
    if enable:
        wanted = {k.strip() for k in enable if k.strip()}
        known = by_key(tower)
        unknown = wanted - known.keys()
        if unknown:
            raise ValueError(f"unknown technique(s) for {tower}: {sorted(unknown)}; "
                             f"available: {sorted(known)}")
        chosen = [t for t in all_techs if t.key in wanted]  # preserve ladder order
        groups: dict[str, list[str]] = {}
        for t in chosen:
            if t.group:
                groups.setdefault(t.group, []).append(t.key)
        clash = {g: ks for g, ks in groups.items() if len(ks) > 1}
        if clash:
            raise ValueError(f"mutually-exclusive techniques selected: {clash}; pick one per group")
        return chosen
    if preset == "full":
        return full_stack(tower)
    if preset == "none" or preset is None:
        return []
    raise ValueError(f"unknown preset {preset!r}; expected one of {PRESETS}")
