"""Technique / backend compatibility (Cosmos 3 report §5.3).

The report splits the optimizations across two backends, and this module encodes that split
so a config can only run techniques its backend actually supports:

  * §5.3.1 Plain PyTorch — the modifiable REFERENCE path. Single-GPU latency techniques:
      torch.compile, CUDA-graph replay (reduce-overhead), Reasoner-tower caching. Attention is
      fixed to cuDNN fused attention here (no math backend exists in cosmos_framework), so the
      math-vs-flash "attention" toggle only applies on the vLLM E-ladder (§5.3.3).
  * §5.3.3 vLLM-Omni — the production Generator runtime. Adds, on top of the above:
      Cache-DiT and dynamic FP8 quantization (plus multi-GPU / memory features handled
      elsewhere).

So Cache-DiT and FP8 are vLLM-Omni ONLY — the native-PyTorch waterfall stops at the Reasoner
cache (E0..E4); E5/E6 are vLLM-Omni rungs. This is the whole point of the waterfall (native
PyTorch reference) vs. production-validation (vLLM/vLLM-Omni) distinction (§4 Jobs 1 vs 2).

It also records technique pairs that do NOT compose cleanly, so the harness can warn (§9):
Cache-DiT's data-dependent block-skipping introduces dynamic control flow that breaks static
CUDA-graph capture — the two must be validated together before being trusted (§9).
"""
from __future__ import annotations

# stage_flags key -> (human name, report section that owns it).
TECHNIQUES = {
    "attention": ("attention backend (math/flash SDPA, vLLM E-ladder)", "§5.3.3"),
    "compile": ("torch.compile", "§5.3.1"),
    "cuda_graphs": ("CUDA-graph replay", "§5.3.1"),
    "reasoner_cache": ("Reasoner-tower caching", "§5.3.1"),
    "cache_dit": ("Cache-DiT", "§5.3.3 (vLLM-Omni)"),
    "quantization": ("dynamic FP8 quantization", "§5.3.3 (vLLM-Omni)"),
}

# What each real backend can honor. The native PyTorch reference implements §5.3.1 only;
# vLLM-Omni adds the §5.3.3 techniques. The mock models the full set (it has no engine).
_PYTORCH = frozenset({"attention", "compile", "cuda_graphs", "reasoner_cache"})   # §5.3.1
_VLLM = _PYTORCH | {"cache_dit", "quantization"}                                  # + §5.3.3
BACKEND_TECHNIQUES = {"pytorch": _PYTORCH, "vllm": _VLLM, "mock": _VLLM}

# FP8 needs FP8 tensor-core hardware (compute capability >= 8.9 Ada / 9.0 Hopper / Blackwell)
# and kernels; checked at runtime on the vLLM-Omni box (§9 acceptance).
FP8_MIN_COMPUTE_CAPABILITY = (8, 9)

# Technique pairs that do not compose cleanly — (frozenset(flags) -> why). §9.
INCOMPATIBLE_PAIRS = {
    frozenset({"cache_dit", "cuda_graphs"}):
        "Cache-DiT skips DiT blocks by a runtime residual-similarity threshold — data-dependent "
        "control flow that breaks static CUDA-graph capture (§9). Only combine if the cache "
        "pattern is made static (fixed skip schedule) AND it still yields an additional speedup.",
}


class UnsupportedTechnique(RuntimeError):
    """A config uses a technique the chosen backend cannot run (wrong report section)."""


def unsupported_flags(config, backend: str) -> list[str]:
    """stage_flags on `config` that `backend` cannot honor (empty == fully supported)."""
    allowed = BACKEND_TECHNIQUES.get(backend, _VLLM)
    return [k for k in config.stage_flags if k in TECHNIQUES and k not in allowed]


def supported(config, backend: str) -> bool:
    return not unsupported_flags(config, backend)


def skip_reason(config, backend: str) -> str | None:
    """Human reason a config is skipped on `backend`, or None if it runs."""
    bad = unsupported_flags(config, backend)
    if not bad:
        return None
    parts = [f"{TECHNIQUES[k][0]} ({TECHNIQUES[k][1]})" for k in bad]
    return f"{', '.join(parts)} not available on the {backend} backend"


def validate(config, backend: str) -> None:
    """Raise UnsupportedTechnique if `config` uses a technique `backend` cannot run."""
    reason = skip_reason(config, backend)
    if reason:
        raise UnsupportedTechnique(f"{config.cid}: {reason}")


def conflicts(config) -> list[tuple[frozenset, str]]:
    """Technique pairs present in `config` that do not compose cleanly (§9 warnings)."""
    flags = {k for k in config.stage_flags if k in TECHNIQUES}
    return [(pair, why) for pair, why in INCOMPATIBLE_PAIRS.items() if pair <= flags]


# waterfall id (configs.END_TO_END) that always runs on the production serving stack.
_END_TO_END = "end_to_end"


def resolve_backend(config, run_backend: str) -> str:
    """Which backend a config actually runs on (Cosmos 3 serving split, §5.3.2/§5.3.3).

    The end-to-end (E) waterfall IS the production stack — vLLM Reasoner + vLLM-Omni Generator
    — so it runs on `vllm`, never the native reference. Any config that needs a vLLM-Omni-only
    technique (Cache-DiT / FP8) also routes there. The component waterfalls (R, G) use the
    native-PyTorch reference (§5.3.1). A `mock` dry-run keeps everything modeled."""
    if run_backend == "mock":
        return "mock"
    if config.waterfall == _END_TO_END or unsupported_flags(config, "pytorch"):
        return "vllm"
    return run_backend
