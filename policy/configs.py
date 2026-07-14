"""The optimization configuration matrix (specification_revised.txt §3).

Three waterfalls, all single-GPU, batch size 1:

  Reasoner conditioning waterfall  R0 -> R3   (measures `reasoner_ms`)
  Generator waterfall              G0 -> G4   (measures `generator_prepare_ms` + `denoising_ms`)
  End-to-end cumulative waterfall  E0 -> E6   (measures `total_chunk_ms`)

Attention baseline: the native-PyTorch reference path (R/G) runs on cosmos_framework, whose
attention dispatcher has NO math/SDPA backend — only fused kernels (flash2/flash3/cuDNN/NATTEN).
So the R/G baseline is torch-native cuDNN *fused* attention (flash-class), forced via
I4_ATTN_BACKENDS=cudnn (policy/pytorch_engine.py); there is no separate math baseline or "+Flash"
rung. The math-vs-flash comparison is realizable only on the vLLM E-ladder (E0 TORCH_SDPA -> E1
FLASH_ATTN, policy/serving.py), which is why it is retained there.

Each rung adds ONE technique to the previous one (cumulative). The end-to-end ladder is
exactly the union the spec lists:

    Baseline eager -> Flash Attention -> torch.compile -> CUDA graphs
      -> Reasoner conditioning cache -> Cache-DiT -> FP8 -> Final

Multi-GPU strategies (CFG-Parallel, Ulysses Context-Parallel) are deliberately NOT on
these ladders — the spec runs them as a separate experiment (§3, policy/multigpu.py).

`stage_multipliers` / `stage_flags` drive the two backends:
  - MockPolicyEngine reads `stage_multipliers` to model per-stage latency with no GPU.
  - The real vLLM/vLLM-Omni + eager path reads `stage_flags` (engine/attention/compile
    knobs). Numbers are anchored to the report where it gives them and to the §7 example
    log; the real backend measures wall-clock and overwrites the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from policy.config import (  # single source of truth (experiment.yaml)
    CONFIG,
    GeneratorSampling,
    ReasonerSampling,
)

# ---------------------------------------------------------------------------
# Pipeline stages (specification_revised.txt §3 stage breakdown / §6 fields).
# ---------------------------------------------------------------------------
REASONER = "reasoner"
GENERATOR = "generator"
END_TO_END = "end_to_end"
WATERFALLS = (REASONER, GENERATOR, END_TO_END)

# The six wall-clock stages the spec's stage breakdown reconciles to.
STAGES = (
    "preprocess",            # images, prompt, state, tensor preparation (+ h2d)
    "reasoner_conditioning", # Reasoner/conditioning computation
    "generator_prepare",     # latent + schedule preparation
    "action_denoising",      # complete action diffusion loop
    "action_postprocess",    # action decode, denormalize, clip (+ d2h)
    "transport",             # client/server communication
)

# Fixed action-chunk geometry — the only evaluated task (§1): 32 timesteps x 8 DoF.
ACTION_CHUNK = CONFIG.dataset.action_chunk

# Sampling recipes — from the single config file. The dataclasses are re-exported so callers
# can keep importing the recipes (and their types) from here alongside the ladder.
REASONER_SAMPLING = CONFIG.reasoner_sampling      # Qwen3-VL conditioning decode params
GENERATOR_SAMPLING = CONFIG.generator_sampling    # action-diffusion recipe

# Diffusion steps for one action-denoising trajectory = the sampling recipe's step count.
# Static (§9 "static shapes") so CUDA-graph configs can capture the loop; denoising_step_ms
# is an array of this length.
N_DENOISE_STEPS = GENERATOR_SAMPLING.steps


@dataclass(frozen=True)
class Config:
    """One rung of a waterfall = one cumulative optimization configuration."""
    cid: str                       # "R0" ... "G4" ... "E6"
    waterfall: str                 # REASONER | GENERATOR | END_TO_END
    label: str                     # human label for tables/plots
    index: int                     # position on its ladder (0 = baseline)
    added: str                     # the single technique this rung adds ("" for baseline)
    lossy: bool = False            # True -> quality-gated (Cache-DiT, FP8): §3, §9
    stage_flags: dict = field(default_factory=dict)     # real-backend knobs (eager + vLLM-Omni)
    stage_multipliers: dict = field(default_factory=dict)  # mock: stage -> speedup divisor (>1 faster)
    reasoner_cached: bool = False  # conditioning computed once/observation vs recomputed per step
    note: str = ""


# ---------------------------------------------------------------------------
# Cumulative *technique* effects. Each technique multiplies the eager cost of the
# stage(s) it touches (divisor > 1 == faster). Cumulative = product along the ladder.
#
# Anchors:
#   - Flash/fused attention (E-ladder / vLLM only): attention-bound stages shrink (reasoner
#     ~1.30, denoise ~1.25). The R/G reference path is already cuDNN-fused, so it has no such rung.
#   - torch.compile: kernel fusion + less host overhead (~1.12-1.15).
#   - CUDA graph replay: removes per-launch overhead; big when the loop is many small
#     kernels (denoise ~1.15), modest on the VLM prefill (~1.10). Not double-counted with
#     compile — measured as the *additional* drop over compile (§9).
#   - Reasoner conditioning cache (R3 / E4): conditioning is invariant across the denoising
#     trajectory, so compute it ONCE per observation instead of every step. In the naive
#     baseline the conditioning is recomputed each of N_DENOISE_STEPS steps; caching removes
#     the (N-1)x. Modeled by `reasoner_cached` (see policy/pipeline.py). Must be invalidated
#     per new observation (§3).
#   - Cache-DiT (G4 / E5, lossy): reuse cached DiT block outputs across adjacent steps ->
#     fewer effective step-compute (~1.40 on the denoise loop).
#   - FP8 (G4 / E6, lossy): dynamic FP8 on the dominant denoise compute (~1.30) + lower
#     peak memory.
# ---------------------------------------------------------------------------
_FLASH_REASONER = 1.30
_FLASH_DENOISE = 1.25
_COMPILE_REASONER = 1.15
_COMPILE_DENOISE = 1.12
_COMPILE_PREP = 1.20            # generator_prepare is host-launch heavy -> compile helps
_CUDAGRAPH_REASONER = 1.10
_CUDAGRAPH_DENOISE = 1.15
_CUDAGRAPH_PREP = 1.25
_CACHEDIT_DENOISE = 1.40
_FP8_DENOISE = 1.30


def _mul(d: dict, stage: str, factor: float) -> None:
    d[stage] = round(d.get(stage, 1.0) * factor, 5)


# ---- Reasoner conditioning waterfall: R0 -> R3 ------------------------------------
# cuDNN fused-attention baseline (see module docstring): no math baseline / "+Flash" rung on
# this native-PyTorch reference path — the framework has no math backend.
def _reasoner_ladder() -> list[Config]:
    m: dict = {}   # cumulative multipliers on reasoner_conditioning
    rows = [
        ("R0", "BF16 + cuDNN fused attention", "", {}, False),
        ("R1", "+ torch.compile", "torch.compile",
         {"compile": True}, False),
        ("R2", "+ CUDA graph replay", "cuda-graphs",
         {"cuda_graphs": True}, False),
        ("R3", "+ Reasoner conditioning cache", "reasoner-conditioning-cache",
         {"reasoner_cache": True}, False),
    ]
    out, cached = [], False
    for i, (cid, label, added, flags, lossy) in enumerate(rows):
        if added == "torch.compile":
            _mul(m, "reasoner_conditioning", _COMPILE_REASONER)
        elif added == "cuda-graphs":
            _mul(m, "reasoner_conditioning", _CUDAGRAPH_REASONER)
        elif added == "reasoner-conditioning-cache":
            cached = True
        out.append(Config(cid, REASONER, label, i, added, lossy,
                          stage_flags=dict(flags), stage_multipliers=dict(m),
                          reasoner_cached=cached,
                          note=("compute conditioning once/observation, reuse across "
                                "denoising steps; invalidate per new observation"
                                if added.endswith("cache") else "")))
    return out


# ---- Generator waterfall: G0 -> G4 ------------------------------------------------
# cuDNN fused-attention baseline like the R ladder (no math / "+Flash" rung). Cache-DiT + FP8
# are the lossy vLLM-Omni rungs (§5.3.3), routed off this path by compat.resolve_backend.
def _generator_ladder() -> list[Config]:
    m: dict = {}   # cumulative multipliers on generator_prepare + action_denoising
    rows = [
        ("G0", "BF16 + cuDNN fused attention", "", {}, False),
        ("G1", "+ torch.compile", "torch.compile", {"compile": True}, False),
        ("G2", "+ CUDA graph replay", "cuda-graphs", {"cuda_graphs": True}, False),
        ("G3", "+ Cache-DiT", "cache-dit", {"cache_dit": True}, True),
        ("G4", "+ Dynamic FP8 quantization", "fp8", {"quantization": "fp8"}, True),
    ]
    out = []
    for i, (cid, label, added, flags, lossy) in enumerate(rows):
        if added == "torch.compile":
            _mul(m, "action_denoising", _COMPILE_DENOISE)
            _mul(m, "generator_prepare", _COMPILE_PREP)
        elif added == "cuda-graphs":
            _mul(m, "action_denoising", _CUDAGRAPH_DENOISE)
            _mul(m, "generator_prepare", _CUDAGRAPH_PREP)
        elif added == "cache-dit":
            _mul(m, "action_denoising", _CACHEDIT_DENOISE)
        elif added == "fp8":
            _mul(m, "action_denoising", _FP8_DENOISE)
        out.append(Config(cid, GENERATOR, label, i, added, lossy,
                          stage_flags=dict(flags), stage_multipliers=dict(m)))
    return out


# ---- End-to-end cumulative waterfall: E0 -> E7 ------------------------------------
# The union the spec lists, applied across BOTH towers cumulatively (§3):
#   Baseline eager -> Flash -> compile -> CUDA graphs -> reasoner cond cache
#     -> Cache-DiT -> FP8 -> Final.
def _end_to_end_ladder() -> list[Config]:
    mr: dict = {}   # reasoner_conditioning multipliers
    mg: dict = {}   # generator (prepare + denoise) multipliers
    rows = [
        ("E0", "Baseline eager (math attn / TORCH_SDPA)", "", {}, False),
        ("E1", "+ Flash Attention (FLASH_ATTN)", "flash-attention", {"attention": "flash"}, False),
        ("E2", "+ torch.compile", "torch.compile", {"compile": True}, False),
        ("E3", "+ CUDA graphs", "cuda-graphs", {"cuda_graphs": True}, False),
        ("E4", "+ Reasoner conditioning cache", "reasoner-conditioning-cache",
         {"reasoner_cache": True}, False),
        ("E5", "+ Cache-DiT", "cache-dit", {"cache_dit": True}, True),
        ("E6", "+ FP8", "fp8", {"quantization": "fp8"}, True),
    ]
    out, cached, flags_acc = [], False, {}
    for i, (cid, label, added, flags, lossy) in enumerate(rows):
        flags_acc = {**flags_acc, **flags}
        if added == "flash-attention":
            _mul(mr, "reasoner_conditioning", _FLASH_REASONER)
            _mul(mg, "action_denoising", _FLASH_DENOISE)
        elif added == "torch.compile":
            _mul(mr, "reasoner_conditioning", _COMPILE_REASONER)
            _mul(mg, "action_denoising", _COMPILE_DENOISE)
            _mul(mg, "generator_prepare", _COMPILE_PREP)
        elif added == "cuda-graphs":
            _mul(mr, "reasoner_conditioning", _CUDAGRAPH_REASONER)
            _mul(mg, "action_denoising", _CUDAGRAPH_DENOISE)
            _mul(mg, "generator_prepare", _CUDAGRAPH_PREP)
        elif added == "reasoner-conditioning-cache":
            cached = True
        elif added == "cache-dit":
            _mul(mg, "action_denoising", _CACHEDIT_DENOISE)
        elif added == "fp8":
            _mul(mg, "action_denoising", _FP8_DENOISE)
        label2 = "Final optimized latency" if cid == "E6" else label
        out.append(Config(cid, END_TO_END, label2, i, added, lossy,
                          stage_flags=dict(flags_acc), stage_multipliers={**mr, **mg},
                          reasoner_cached=cached))
    # rename the last rung's cid-facing label already handled; final rung is E6.
    return out


REASONER_LADDER: list[Config] = _reasoner_ladder()
GENERATOR_LADDER: list[Config] = _generator_ladder()
END_TO_END_LADDER: list[Config] = _end_to_end_ladder()

_LADDERS: dict[str, list[Config]] = {
    REASONER: REASONER_LADDER,
    GENERATOR: GENERATOR_LADDER,
    END_TO_END: END_TO_END_LADDER,
}


def ladder(waterfall: str) -> list[Config]:
    if waterfall not in _LADDERS:
        raise ValueError(f"unknown waterfall {waterfall!r}; expected one of {WATERFALLS}")
    return _LADDERS[waterfall]


def all_configs() -> list[Config]:
    """Every rung of every ladder — the full single-GPU matrix run by run_matrix.py.

    Order: R0-R3, G0-G4, E0-E6 (the combined end-to-end configurations)."""
    return [*REASONER_LADDER, *GENERATOR_LADDER, *END_TO_END_LADDER]


def config_by_id(cid: str) -> Config:
    for c in all_configs():
        if c.cid == cid:
            return c
    raise ValueError(f"unknown configuration {cid!r}; known: {[c.cid for c in all_configs()]}")


def baseline_id(waterfall: str) -> str:
    return ladder(waterfall)[0].cid


def final_id(waterfall: str) -> str:
    return ladder(waterfall)[-1].cid


def is_quality_gated(cid: str) -> bool:
    """Cache-DiT / FP8 rungs (and any E-rung that has enabled them) are quality-gated."""
    return config_by_id(cid).lossy


def matrix_summary() -> str:
    lines = []
    for wf in WATERFALLS:
        lines.append(f"[{wf}]")
        for c in ladder(wf):
            tag = "  (lossy → quality-gate)" if c.lossy else ""
            lines.append(f"  {c.cid:<4} {c.label}{tag}")
    return "\n".join(lines)
