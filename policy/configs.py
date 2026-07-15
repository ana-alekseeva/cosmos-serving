"""The optimization configuration matrix.

Two waterfalls, all single-GPU, batch size 1:
  Native-PyTorch reference  P0 -> P3   (R+G merged — one MoT)
  End-to-end cumulative     E0 -> E4   (vLLM stack)

Each rung cumulatively adds one technique. `stage_multipliers` drives the mock backend;
`stage_flags` drives the real vLLM/eager backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from policy.config import (  # single source of truth (experiment.yaml)
    CONFIG,
    GeneratorSampling,
    ReasonerSampling,
)

# R and G ladders measured the SAME single MoT inference -> merged into ONE native ladder P.
NATIVE = "native"
END_TO_END = "end_to_end"
WATERFALLS = (NATIVE, END_TO_END)

# The six wall-clock stages the spec's stage breakdown reconciles to.
STAGES = (
    "preprocess",            # images, prompt, state, tensor preparation (+ h2d)
    "reasoner_conditioning", # Reasoner/conditioning computation
    "generator_prepare",     # latent + schedule preparation
    "action_denoising",      # complete action diffusion loop
    "action_postprocess",    # action decode, denormalize, clip (+ d2h)
    "transport",             # client/server communication
)

# Fixed action-chunk geometry — the only evaluated task: 32 timesteps x 8 DoF.
ACTION_CHUNK = CONFIG.dataset.action_chunk

REASONER_SAMPLING = CONFIG.reasoner_sampling      # Qwen3-VL conditioning decode params
GENERATOR_SAMPLING = CONFIG.generator_sampling    # action-diffusion recipe

# Static shape so CUDA-graph configs can capture the loop.
N_DENOISE_STEPS = GENERATOR_SAMPLING.steps


@dataclass(frozen=True)
class Config:
    """One rung of a waterfall = one cumulative optimization configuration."""
    cid: str                       # "P0" ... "P3" ... "E4"
    waterfall: str                 # NATIVE | END_TO_END
    label: str                     # human label for tables/plots
    index: int                     # position on its ladder (0 = baseline)
    added: str                     # the single technique this rung adds ("" for baseline)
    lossy: bool = False            # True -> quality-gated (Cache-DiT, FP8)
    stage_flags: dict = field(default_factory=dict)     # real-backend knobs (eager + vLLM-Omni)
    stage_multipliers: dict = field(default_factory=dict)  # mock: stage -> speedup divisor (>1 faster)
    reasoner_cached: bool = False  # conditioning computed once/observation vs recomputed per step
    note: str = ""


# Cumulative technique speedup divisors (>1 == faster); mock-model anchors only.
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


# ---- Native-PyTorch reference waterfall: P0 -> P3 ---------------------------------
# cuDNN fused-attention baseline (framework has no math backend, so no "+Flash" rung).
def _native_ladder() -> list[Config]:
    mr: dict = {}   # reasoner_conditioning multipliers
    mg: dict = {}   # generator (prepare + denoise) multipliers
    rows = [
        ("P0", "BF16 + cuDNN fused attention", "", {}, False),
        ("P1", "+ torch.compile", "torch.compile", {"compile": True}, False),
        ("P2", "+ CUDA graph replay", "cuda-graphs", {"cuda_graphs": True}, False),
        ("P3", "+ Reasoner conditioning cache", "reasoner-conditioning-cache",
         {"reasoner_cache": True}, False),
    ]
    out, cached, flags_acc = [], False, {}
    for i, (cid, label, added, flags, lossy) in enumerate(rows):
        flags_acc = {**flags_acc, **flags}          # cumulative: each rung keeps the prior knobs
        if added == "torch.compile":
            _mul(mr, "reasoner_conditioning", _COMPILE_REASONER)
            _mul(mg, "action_denoising", _COMPILE_DENOISE)
            _mul(mg, "generator_prepare", _COMPILE_PREP)
        elif added == "cuda-graphs":
            _mul(mr, "reasoner_conditioning", _CUDAGRAPH_REASONER)
            _mul(mg, "action_denoising", _CUDAGRAPH_DENOISE)
            _mul(mg, "generator_prepare", _CUDAGRAPH_PREP)
        elif added == "reasoner-conditioning-cache":
            cached = True
        out.append(Config(cid, NATIVE, label, i, added, lossy,
                          stage_flags=dict(flags_acc), stage_multipliers={**mr, **mg},
                          reasoner_cached=cached,
                          note=("compute conditioning once/observation, reuse across "
                                "denoising steps; invalidate per new observation"
                                if added.endswith("cache") else "")))
    return out


# ---- End-to-end cumulative waterfall: E0 -> E4 ------------------------------------
# Cache-DiT and reasoner-conditioning-cache rungs were REMOVED after H100 Job 2 (2026-07-15):
# Cache-DiT never activates at W=4 and bypasses the compiled transformer; conditioning cache
# is not implementable on stock vllm-omni. The engine still supports the cache_dit flag.
def _end_to_end_ladder() -> list[Config]:
    mr: dict = {}   # reasoner_conditioning multipliers
    mg: dict = {}   # generator (prepare + denoise) multipliers
    rows = [
        ("E0", "Baseline eager (math attn / TORCH_SDPA)", "", {}, False),
        ("E1", "+ Flash Attention (FLASH_ATTN)", "flash-attention", {"attention": "flash"}, False),
        ("E2", "+ torch.compile", "torch.compile", {"compile": True}, False),
        ("E3", "+ CUDA graphs", "cuda-graphs", {"cuda_graphs": True}, False),
        ("E4", "+ FP8", "fp8", {"quantization": "fp8"}, True),
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
        label2 = "Final optimized latency" if cid == rows[-1][0] else label
        out.append(Config(cid, END_TO_END, label2, i, added, lossy,
                          stage_flags=dict(flags_acc), stage_multipliers={**mr, **mg},
                          reasoner_cached=cached))
    return out


NATIVE_LADDER: list[Config] = _native_ladder()
END_TO_END_LADDER: list[Config] = _end_to_end_ladder()

_LADDERS: dict[str, list[Config]] = {
    NATIVE: NATIVE_LADDER,
    END_TO_END: END_TO_END_LADDER,
}


def ladder(waterfall: str) -> list[Config]:
    if waterfall not in _LADDERS:
        raise ValueError(f"unknown waterfall {waterfall!r}; expected one of {WATERFALLS}")
    return _LADDERS[waterfall]


def all_configs() -> list[Config]:
    """Every rung of every ladder — the full single-GPU matrix run by run_matrix.py.

    Order: P0-P3 (native PyTorch), E0-E4 (the combined end-to-end vLLM configurations)."""
    return [*NATIVE_LADDER, *END_TO_END_LADDER]


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
