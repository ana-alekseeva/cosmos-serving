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
    default_on: bool = False   # True -> vLLM ships this ENABLED; naive baseline disables it. The
                               # contiguous default_on prefix of the ladder == the stock-vLLM config.


# ---------------------------------------------------------------------------
# Reasoner ladder (Part 1a) — canonical cumulative order.
# op names: A (latency), B (decode), C (throughput), D (multimodal / robot).
# ---------------------------------------------------------------------------
# The naive vLLM baseline DISABLES each default-on knob (--enforce-eager, SDPA attention,
# --no-enable-prefix-caching, --no-enable-chunked-prefill, --max-num-seqs 1); each technique
# re-enables exactly one. See bench/serving.py::build_command. Mock speedups are anchored to
# the real H200 measurements where we have them (cuda-graphs, fp8).
REASONER_TECHNIQUES: list[Technique] = [
    Technique("cuda-graphs", "torch.compile + CUDA graphs", REASONER, False, "A",
              {"vllm": {"enforce_eager": False}},
              {"A": 3.35, "B": 3.40, "C": 3.08, "D": 2.51, "E": 3.12, "F": 1.16},
              default_on=True),
    Technique("flash-attn", "FlashAttention (vs SDPA)", REASONER, False, "B",
              {"env": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"}},
              {"A": 1.20, "B": 1.60, "C": 1.40, "D": 1.50, "E": 1.40, "F": 1.50},
              default_on=True),
    Technique("prefix-caching", "prefix caching", REASONER, False, "C",
              {"vllm": {"enable_prefix_caching": True}},
              {"A": 1.02, "B": 1.02, "C": 1.05, "D": 1.02, "E": 1.02, "F": 1.05},
              default_on=True),
    Technique("chunked-prefill", "chunked prefill", REASONER, False, "D",
              {"vllm": {"enable_chunked_prefill": True}},
              {"A": 1.05, "B": 1.05, "C": 1.10, "D": 1.20, "E": 1.15, "F": 1.20},
              default_on=True),
    Technique("continuous-batching", "continuous batching (max-num-seqs)", REASONER, False, "C",
              {"vllm": {"max_num_seqs": ">1"}},
              {"A": 1.00, "B": 1.00, "C": 3.00, "D": 1.00, "E": 1.00, "F": 2.50},
              category="throughput", default_on=True),   # vLLM batches by default; part of "stock vLLM"
    Technique("fp8", "FP8 / NVFP4 quantization", REASONER, True, "F",
              {"vllm": {"quantization": "fp8"}},
              {"A": 1.36, "B": 1.38, "C": 1.22, "D": 1.26, "E": 1.23, "F": 1.36}),
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
    # include eager-only reasoner techniques (e.g. kv-cache) so --enable/--ablate accept them
    techs = list(techniques_for(tower))
    if tower == REASONER:
        seen = {t.key for t in techs}
        techs += [t for t in EAGER_REASONER_TECHNIQUES if t.key not in seen]
    return {t.key: t for t in techs}


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


# Eager reference-path ladder (cosmos-framework / HF Transformers) — the FUNDAMENTAL
# techniques vLLM bakes in and can't toggle. KV cache is the headline (huge on the
# long-output OP-B). Used only with --backend eager, on single-request OPs (A/B/D/E).
EAGER_REASONER_TECHNIQUES: list[Technique] = [
    Technique("kv-cache", "KV cache (use_cache)", REASONER, False, "B",
              {"hf": {"use_cache": True}},
              {"A": 2.50, "B": 6.00, "C": 2.00, "D": 1.50, "E": 1.50, "F": 1.60}),
    Technique("cuda-graphs", "torch.compile / CUDA graphs", REASONER, False, "A",
              {"hf": {"compile": True}},
              {"A": 1.60, "B": 1.15, "C": 1.10, "D": 1.10, "E": 1.15, "F": 1.10}),
    Technique("flash-attn", "FlashAttention (attn_implementation)", REASONER, False, "B",
              {"hf": {"attn_implementation": "flash_attention_2"}},
              {"A": 1.20, "B": 1.50, "C": 1.30, "D": 1.40, "E": 1.35, "F": 1.40}),
    Technique("fp8", "FP8 quantization", REASONER, True, "F",
              {"hf": {"quantization": "fp8"}},
              {"A": 1.20, "B": 1.50, "C": 1.40, "D": 1.30, "E": 1.30, "F": 1.35}),
]


def ablation_ladder(tower: str, backend: str = "vllm") -> list[Technique]:
    """Cumulative waterfall order. The eager backend uses the reference-path ladder
    (the fundamentals vLLM can't toggle); everyone else uses the full stack minus
    memory-only techniques (which never reduce latency). Throughput techniques
    (e.g. continuous batching) STAY in the ladder but are rendered per-OP: they only
    appear on high-concurrency OPs, where they act (see AblationResult.marginal_rows)."""
    if tower == REASONER and backend == "eager":
        return list(EAGER_REASONER_TECHNIQUES)
    return [t for t in full_stack(tower) if t.category != "memory"]


def resolve(tower: str, *, preset: str | None = None,
            enable: list[str] | None = None) -> list[Technique]:
    """Enabled techniques (in ladder order) for a preset or explicit subset."""
    all_techs = list(by_key(tower).values())   # vLLM ladder + eager-only reasoner keys
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
