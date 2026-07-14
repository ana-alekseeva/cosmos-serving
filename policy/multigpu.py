"""Separate multi-GPU experiment (specification_revised.txt §3).

NOT part of the primary single-GPU waterfall — the spec is explicit: "Do not mix different
GPU counts in the primary waterfall." This is a standalone job that runs:

  * CFG parallelism   — only if the action pipeline uses classifier-free guidance.
  * Ulysses context parallelism.

and compares each against the BEST single-GPU configuration (the final E6). CPU offload,
HSDP, and VAE patch parallelism are deliberately excluded (memory-oriented / inapplicable
to action output, §3).

The mock models the 2-GPU per-step latency reduction; the real path launches vLLM-Omni with
`--tensor-parallel-size 2` + the parallel strategy (VERIFY flags on-box).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from policy.config import CONFIG
from policy.configs import END_TO_END, GENERATOR_SAMPLING, final_id
from policy.dataset import load_manifest, tile_to
from policy.pipeline import make_engine
from policy.configs import config_by_id

# Whether the action pipeline uses CFG. If False, CFG parallelism is skipped (§3). Derived
# from the generator sampling recipe: the DROID policy samples with full-range CFG Null
# (guidance=3), so CFG is active and the CFG-Parallel experiment applies.
ACTION_PIPELINE_USES_CFG = GENERATOR_SAMPLING.uses_cfg

# Modeled 2-GPU per-step denoise reduction vs the best single-GPU config (mock).
#   CFG-Parallel: cond/uncond on 2 GPUs -> "nearly halves per-step latency".
#   Ulysses Context-Parallel: sequence-sharded attention -> ~1.7x.
_STRATEGY_DENOISE_SPEEDUP = {"cfg-parallel": 1.85, "ulysses-context-parallel": 1.70}


@dataclass
class MultiGpuStrategy:
    key: str
    label: str
    gpus: int
    requires_cfg: bool
    denoise_speedup: float


STRATEGIES = [
    MultiGpuStrategy("cfg-parallel", "CFG-Parallel (2 GPU)", 2, True,
                     _STRATEGY_DENOISE_SPEEDUP["cfg-parallel"]),
    MultiGpuStrategy("ulysses-context-parallel", "Ulysses Context-Parallel (2 GPU)", 2, False,
                     _STRATEGY_DENOISE_SPEEDUP["ulysses-context-parallel"]),
]


def _p50_chunk(engine, requests) -> float:
    vals = [engine.run_request(r).total_chunk_ms for r in requests]
    return float(np.median(vals))


def run_multigpu(*, backend: str = "mock", manifest: str = "policy/mock/manifest.json",
                 model: str | None = None, out_dir: str | Path = "results") -> dict:
    """Compare CFG-/Ulysses-parallel against the best single-GPU config (final E6)."""
    requests = tile_to(load_manifest(manifest), CONFIG.dataset.replay_size)
    best_single = config_by_id(final_id(END_TO_END))          # best single-GPU baseline (§3)

    single_engine = make_engine(backend, best_single, model=model)
    single_engine.prepare()
    try:
        single_p50 = _p50_chunk(single_engine, requests)
    finally:
        single_engine.close()

    rows = []
    for strat in STRATEGIES:
        if strat.requires_cfg and not ACTION_PIPELINE_USES_CFG:
            rows.append({"strategy": strat.key, "label": strat.label, "gpus": strat.gpus,
                         "skipped": True, "reason": "action pipeline does not use CFG (§3)"})
            continue
        p50 = _multigpu_p50(backend, best_single, strat, requests, model)
        rows.append({
            "strategy": strat.key, "label": strat.label, "gpus": strat.gpus,
            "skipped": False,
            "single_gpu_p50_chunk_ms": round(single_p50, 3),
            "multigpu_p50_chunk_ms": round(p50, 3),
            "speedup_vs_best_single_gpu": round(single_p50 / p50, 3) if p50 else None,
            "note": "scaling result — bar = adding a 2nd GPU, not a per-GPU algorithmic win",
        })

    result = {
        "experiment": "multi_gpu",
        "best_single_gpu_config": best_single.cid,
        "action_pipeline_uses_cfg": ACTION_PIPELINE_USES_CFG,
        "single_gpu_p50_chunk_ms": round(single_p50, 3),
        "strategies": rows,
    }
    out = Path(out_dir) / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    (out / "multigpu.json").write_text(json.dumps(result, indent=2))
    return result


def _multigpu_p50(backend: str, best_single, strat: MultiGpuStrategy, requests, model) -> float:
    if backend != "mock":
        # VERIFY: launch vLLM-Omni --tensor-parallel-size 2 with the parallel strategy.
        raise NotImplementedError(
            f"Real {strat.key} runs on 2 GPUs via vLLM-Omni --tensor-parallel-size 2; "
            "the mock models the per-step reduction.")
    # Model the 2-GPU config as the best single-GPU config with denoise sped up further.
    from dataclasses import replace
    mult = dict(best_single.stage_multipliers)
    mult["action_denoising"] = round(mult.get("action_denoising", 1.0) * strat.denoise_speedup, 5)
    scaled = replace(best_single, cid=f"{best_single.cid}+{strat.key}", stage_multipliers=mult)
    engine = make_engine(backend, scaled, model=model)
    engine.prepare()
    try:
        return _p50_chunk(engine, requests)
    finally:
        engine.close()
