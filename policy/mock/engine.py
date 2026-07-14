"""Modeled latency engine (MOCK backend — specification_revised.txt §6).

`MockPolicyEngine` models each §6 stage from the config's `stage_multipliers` so the whole
harness (logs, waterfalls, aggregation) runs with NO GPU / NO model. Anchored to the §7
example log (server_total ~175 ms at the generator baseline). Deterministic per request
seed -> reproducible p50/p90/p99. The real path lives in policy/pipeline.py.

Stage model (eager baseline, once per observation). Per-stage costs are anchored to the §7
example log; the denoising loop runs the Cosmos3-Nano-Policy-DROID sampling recipe
(configs.GENERATOR_SAMPLING: steps=4, guidance=3, shift=5, full-range CFG Null), so it is 4
steps — not the 20 of the spec's illustrative example row:
    preprocess 7.1 · h2d 1.2 · reasoner(single) 45.8 · generator_prepare 2.4
    · denoise 4 x 5.8 = 23.2 · postprocess 1.8 · d2h 0.4 · transport ~9
    => server_total ~81.9, chunk_total ~90.9
The per-step cost (5.8 ms) folds in the full-range CFG two-forward cost; re-anchor it (and
the stage costs above) from an on-box eager trace before trusting absolute numbers.

Reasoner conditioning cache (P3 / E4): the conditioning is invariant across the denoising
trajectory, so the naive baseline recomputes it every one of N_DENOISE_STEPS steps; caching
computes it ONCE per observation. Modeled by REASONER_RECOMPUTE_FRACTION (=1.0, full
recompute) and toggled by `config.reasoner_cached`.
"""
from __future__ import annotations

import hashlib
import random
import zlib

from policy.config import CONFIG
from policy.configs import N_DENOISE_STEPS, Config
from policy.dataset import DroidRequest
from policy.measure import LatencyRecord

# ---- Eager-baseline stage costs (ms), anchored to the §7 example log --------------
BASE = {
    "preprocess": 7.1,
    "h2d": 1.2,
    "reasoner_single": 45.8,       # one full conditioning pass (VLM + projection)
    "generator_prepare": 2.4,
    "denoise_per_step": 5.8,       # x N_DENOISE_STEPS
    "postprocess": 1.8,
    "d2h": 0.4,
    "transport": 9.0,
}
BASELINE_PEAK_MEMORY_MB = 43120.0
# Fraction of the reasoner conditioning that the naive path recomputes per denoising step.
# 1.0 == the whole conditioning is recomputed every step (the literal §3 baseline that the
# reasoner-conditioning-cache removes). Tune from the on-box eager trace.
REASONER_RECOMPUTE_FRACTION = 1.0
# Modeled action drift for the lossy techniques (Cache-DiT, FP8), vs the eager output.
# Below the gate threshold -> "passed" (the RoboLab subset makes the real call, §5/§9).
# _LOSSY_DRIFT is a mock-simulator anchor (stays in code); the gate THRESHOLD is a run
# parameter from the single config file (config/experiment.yaml -> quality_gate).
_LOSSY_DRIFT = {"cache_dit": 0.006, "quantization": 0.011}   # normalized action-MSE
QUALITY_GATE_THRESHOLD = CONFIG.quality_gate.action_mse_threshold


def _reasoner_ms(config: Config, single_eff: float) -> float:
    """Effective reasoner conditioning time. Recomputed per step unless the reasoner-conditioning
    cache is on (P3 / E4), which holds it at the single-conditioning cost."""
    cached = config.reasoner_cached
    if cached:
        return single_eff
    f = REASONER_RECOMPUTE_FRACTION
    return single_eff * (1.0 - f) + single_eff * f * N_DENOISE_STEPS


def _peak_memory_mb(config: Config) -> float:
    mb = BASELINE_PEAK_MEMORY_MB
    flags = config.stage_flags
    if flags.get("quantization") == "fp8":
        mb *= 0.62                                  # FP8 weights/activations -> big VRAM cut
    if flags.get("cache_dit"):
        mb *= 1.02                                  # DiT block-output cache costs a little VRAM
    if config.reasoner_cached:
        mb *= 0.99                                  # no per-step recompute scratch buffers
    return mb


def _checksum(req: DroidRequest, config: Config) -> str:
    """Deterministic per-request output checksum. Lossless configs produce the SAME checksum
    as the eager baseline (numerical equivalence); lossy configs (Cache-DiT/FP8) deviate
    deterministically so drift is visible in the logs."""
    salt = "|".join(sorted(k for k in ("cache_dit", "quantization") if config.stage_flags.get(k)))
    h = hashlib.sha1(f"{req.seed}:{salt}".encode()).hexdigest()
    return h[:16]


def _quality_gate(config: Config) -> tuple[str, float]:
    if not config.lossy:
        return "passed", 0.0                        # lossless -> exact match
    drift = 0.0
    for k, d in _LOSSY_DRIFT.items():
        if config.stage_flags.get(k):
            drift += d
    return ("passed" if drift <= QUALITY_GATE_THRESHOLD else "failed"), round(drift, 4)


class MockPolicyEngine:
    backend = "mock"

    def __init__(self, config: Config, *, model: str | None = None, **_):
        self.config = config
        self.model = model or "nvidia/Cosmos3-Nano-Policy-DROID"
        self._gate, self._drift = _quality_gate(config)

    def prepare(self) -> None:
        """Load model from local storage (no-op for the mock)."""

    def run_request(self, req: DroidRequest) -> LatencyRecord:
        c = self.config
        mult = c.stage_multipliers
        # Process-STABLE seed (not builtin hash(): that is PYTHONHASHSEED-randomized, which
        # would make two runs of the same config in different subprocesses disagree — §10
        # requires reproducible logs).
        rng = random.Random(req.seed ^ zlib.crc32(c.cid.encode()))

        def stage(base: float, key: str) -> float:
            # jitter is deterministic in the request seed (reproducible percentiles, §10)
            jitter = 1.0 + (rng.random() - 0.5) * 0.06           # +/-3%
            return base / mult.get(key, 1.0) * jitter

        preprocess = stage(BASE["preprocess"], "preprocess")
        h2d = stage(BASE["h2d"], "h2d")
        single_eff = stage(BASE["reasoner_single"], "reasoner_conditioning")
        reasoner = _reasoner_ms(c, single_eff)
        gen_prep = stage(BASE["generator_prepare"], "generator_prepare")
        per_step = BASE["denoise_per_step"] / mult.get("action_denoising", 1.0)
        steps = [per_step * (1.0 + (rng.random() - 0.5) * 0.06) for _ in range(N_DENOISE_STEPS)]
        denoising = sum(steps)
        postprocess = stage(BASE["postprocess"], "postprocess")
        d2h = stage(BASE["d2h"], "d2h")
        transport = stage(BASE["transport"], "transport")

        server = preprocess + h2d + reasoner + gen_prep + denoising + postprocess + d2h
        chunk_total = server + transport
        first_action = chunk_total          # non-streaming: whole 32-chunk ready together

        return LatencyRecord(
            request_id=req.request_id, task=req.task, episode_id=req.episode_id,
            preprocess_ms=preprocess, h2d_ms=h2d, reasoner_ms=reasoner,
            generator_prepare_ms=gen_prep, denoising_ms=denoising,
            denoising_step_ms=steps, postprocess_ms=postprocess, d2h_ms=d2h,
            server_ms=server, transport_ms=transport,
            first_action_ms=first_action, total_chunk_ms=chunk_total,
            peak_memory_mb=_peak_memory_mb(c),
            output_checksum=_checksum(req, c), quality_gate=self._gate,
        )

    def close(self) -> None:
        pass
