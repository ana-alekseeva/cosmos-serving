"""Technique registry — the single source of truth for the `optimize` command.

Scope note (why this is Generator-only): the Cosmos 3 report (§5.3.2) serves the
Reasoner with vLLM / TensorRT-LLM by *reusing upstream Qwen3-VL out of the box* —
it presents no reasoner technique ablation. So the Reasoner has no technique ladder
here; it is measured as a stock-vLLM concurrency/shape sweep (bench.sweep). Every
quantified serving optimization in the report is on the **Generator** (§5.3.1/§5.3.3):
torch.compile+CUDA graphs (30-60% on T2I), reasoner-tower caching, Cache-DiT, FP8,
VAE-Patch-Parallel, CFG-/Context-Parallel, and request batching (Table 9). Those are
the toggles below.

`mock_speedups` is used ONLY by bench.drivers.MockEngine so the ablation and plots
run without a GPU. The values are anchored to the report's stated numbers where it
gives them (see comments). On the H200 the real backend reads `engine_flags` and
measures wall-clock instead (see specification.md §5-§6).
"""
from __future__ import annotations

from dataclasses import dataclass

REASONER = "reasoner"
GENERATOR = "generator"
TOWERS = (REASONER, GENERATOR)


@dataclass(frozen=True)
class Technique:
    key: str                 # CLI token, e.g. "cuda-graphs"
    label: str               # human label for tables/plots
    tower: str               # REASONER | GENERATOR
    lossy: bool              # True -> triggers the quality guard
    regime: str              # primary operating point where it acts (UX/doc)
    engine_flags: dict       # real-backend flags (wired on the GPU)
    mock_speedups: dict      # op_name -> latency divisor (>1 = faster); mock only
    note: str = ""
    scaling: bool = False      # True -> uses extra GPUs (CFG-/Context-Parallel): a scaling win, not per-GPU
    category: str = "latency"  # latency | scaling | throughput | memory
    group: str = ""            # non-empty -> mutually-exclusive alternatives (one per group on N GPUs)


# ---------------------------------------------------------------------------
# Generator ladder (the report's quantified serving story) — cumulative order.
# op names: t2i-1024 / t2v-256 / t2v-480 / t2v-720 / i2v-480.
#
# Anchors to the report:
#   - cuda-graphs: "CUDA Graphs on T2I generation yielded 30% to 60% speedups"
#     (§5.3.1). Modeled as a ~45% latency drop on T2I (1.82x), diminishing sharply on
#     video, where larger kernels make host launch overhead a small fraction.
#   - cfg-parallel: "nearly halves the per-step latency" (§5.3.1) -> ~1.85x, 2-GPU.
#   - batching: Table 9 (T2V, 189 frames) is a THROUGHPUT result, not a latency one, so
#     it is excluded from the latency waterfall (category="throughput") and reproduced
#     separately by the batching sweep (bench.sweep). Per-OP gains live on the OP.
# ---------------------------------------------------------------------------
GENERATOR_TECHNIQUES: list[Technique] = [
    Technique("reasoner-cache", "reasoner-tower output caching", GENERATOR, False, "all",
              {"cosmos": {"reasoner_cache": True}},
              {"t2i-1024": 1.12, "t2v-256": 1.10, "t2v-480": 1.09, "t2v-720": 1.08, "i2v-480": 1.08},
              note="Conditioning is invariant across denoise steps -> compute once, reuse."),
    Technique("cuda-graphs", "torch.compile / CUDA graphs", GENERATOR, False, "t2i-1024",
              {"cosmos": {"cuda_graphs": True}},
              {"t2i-1024": 1.82, "t2v-256": 1.10, "t2v-480": 1.05, "t2v-720": 1.03, "i2v-480": 1.06},
              note="Report: 30-60% on T2I; host-launch-bound, so it fades on longer video."),
    Technique("cache-dit", "Cache-DiT", GENERATOR, True, "t2v-720",
              {"vllm_omni": {"cache_dit": True}},
              {"t2i-1024": 1.15, "t2v-256": 1.35, "t2v-480": 1.45, "t2v-720": 1.55, "i2v-480": 1.50},
              note="Training-free reuse of cached block outputs across adjacent steps; more steps -> more win."),
    Technique("fp8", "FP8 quantization", GENERATOR, True, "t2v-720",
              {"vllm_omni": {"quantization": "fp8"}},
              {"t2i-1024": 1.40, "t2v-256": 1.45, "t2v-480": 1.48, "t2v-720": 1.50, "i2v-480": 1.50},
              note="Dynamic FP8 on dominant compute; memory-bound denoise benefits most."),
    Technique("vae-patch", "VAE-Patch-Parallel", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"vae_patch_parallel": True}},
              {"t2i-1024": 1.05, "t2v-256": 1.06, "t2v-480": 1.12, "t2v-720": 1.18, "i2v-480": 1.12},
              note="Tiles the VAE decode across ranks; shrinks the decode tail at high resolution."),
    Technique("cfg-parallel", "CFG-Parallel (2 GPU)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"cfg_parallel": True, "gpus": 2}},
              {"t2i-1024": 1.80, "t2v-256": 1.85, "t2v-480": 1.86, "t2v-720": 1.88, "i2v-480": 1.86},
              note="Report: cond/uncond on 2 GPUs 'nearly halves per-step latency'. A scaling win (2nd GPU).",
              scaling=True, category="scaling", group="distributed"),
    Technique("context-parallel", "Ulysses Context-Parallel (2 GPU)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"context_parallel": True, "ulysses_degree": 2, "gpus": 2}},
              {"t2i-1024": 1.55, "t2v-256": 1.65, "t2v-480": 1.68, "t2v-720": 1.70, "i2v-480": 1.68},
              note="Alternative 2-GPU strategy to CFG-Parallel (same group; not stacked on 2 GPUs).",
              scaling=True, category="scaling", group="distributed"),
    Technique("batching", "request batching (seq-packing)", GENERATOR, False, "t2v-256",
              {"cosmos": {"batching": True}},
              {"t2i-1024": 1.0, "t2v-256": 1.0, "t2v-480": 1.0, "t2v-720": 1.0, "i2v-480": 1.0},
              note="Throughput only (Table 9); no per-clip latency benefit. See the batching sweep.",
              category="throughput"),
    Technique("hsdp", "HSDP (FSDP2 weight sharding)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"hsdp": True}},
              {"t2i-1024": 1.0, "t2v-256": 1.0, "t2v-480": 1.0, "t2v-720": 1.0, "i2v-480": 1.0},
              note="Memory reduction (peak VRAM); neutral on single-request latency.",
              category="memory"),
    Technique("cpu-offload", "CPU offload (layer-wise)", GENERATOR, False, "t2v-720",
              {"vllm_omni": {"enable_layerwise_offload": True}},
              {"t2i-1024": 0.80, "t2v-256": 0.80, "t2v-480": 0.80, "t2v-720": 0.80, "i2v-480": 0.80},
              note="Memory reduction; ADDS latency (PCIe transfers). Only for memory-constrained GPUs.",
              category="memory"),
]

_LADDERS: dict[str, list[Technique]] = {
    REASONER: [],                       # no technique ladder — stock vLLM concurrency sweep (bench.sweep)
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
    """Cumulative *latency* waterfall order: the full stack minus techniques that don't
    reduce single-request latency — memory-reduction (HSDP, CPU offload) and throughput
    (batching, reproduced by the batching sweep instead)."""
    return [t for t in full_stack(tower) if t.category not in ("memory", "throughput")]


def resolve(tower: str, *, preset: str | None = None,
            enable: list[str] | None = None) -> list[Technique]:
    """Enabled techniques (in ladder order) for a preset or explicit subset."""
    all_techs = list(techniques_for(tower))
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
