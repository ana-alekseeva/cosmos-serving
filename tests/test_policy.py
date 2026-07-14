"""Sanity tests for the Cosmos3-Nano-Policy-DROID harness — `uv run pytest`.

Tests LOGIC and INVARIANTS (waterfall monotonicity, the §7 log schema, reproducibility,
the §8 drift check, the §10 acceptance checks, the quality gate), not the mock's modeled
constants (those are replaced by real vLLM/vLLM-Omni measurements on the GPU) nor the
matplotlib rendering.
"""
from pathlib import Path

import pytest

from policy.aggregate import aggregate, build_stage_breakdown, load_results
from policy.configs import (
    ACTION_CHUNK,
    END_TO_END,
    END_TO_END_LADDER,
    GENERATOR,
    GENERATOR_LADDER,
    GENERATOR_SAMPLING,
    N_DENOISE_STEPS,
    REASONER_SAMPLING,
    REASONER,
    REASONER_LADDER,
    STAGES,
    all_configs,
    config_by_id,
    ladder,
)
from policy import compat
from policy.compat import UnsupportedTechnique
from policy.dataset import quality_subset
from policy.experiment import Experiment
from policy.matrix import run_matrix
from policy.measure import LatencyRecord
from policy.mock.replay import build_mock_replay, write_mock_manifest
from policy.pipeline import make_engine
from policy import robolab


# -- Config matrix (§3) -----------------------------------------------------------
def test_ladders_match_spec():
    assert [c.cid for c in REASONER_LADDER] == ["R0", "R1", "R2", "R3"]
    assert [c.cid for c in GENERATOR_LADDER] == ["G0", "G1", "G2", "G3", "G4"]
    assert [c.cid for c in END_TO_END_LADDER] == ["E0", "E1", "E2", "E3", "E4", "E5", "E6"]
    # baselines add no technique (R/G start from the cuDNN fused-attention baseline)
    for lad in (REASONER_LADDER, GENERATOR_LADDER, END_TO_END_LADDER):
        assert lad[0].added == "" and not lad[0].lossy


def test_only_cachedit_and_fp8_are_lossy():
    lossy = {c.cid for c in all_configs() if c.lossy}
    # G3 Cache-DiT, G4 FP8, and every E-rung that has enabled them (E5, E6)
    assert lossy == {"G3", "G4", "E5", "E6"}


def test_end_to_end_ladder_is_the_union_spec_lists():
    added = [c.added for c in END_TO_END_LADDER[1:]]
    assert added == ["flash-attention", "torch.compile", "cuda-graphs",
                     "reasoner-conditioning-cache", "cache-dit", "fp8"]


# -- Replay dataset + quality subset (§5) -----------------------------------------
def test_replay_set_is_fixed_and_deterministic():
    a = build_mock_replay(256, seed=20260713)
    b = build_mock_replay(256, seed=20260713)
    assert len(a) == 256
    assert [r.seed for r in a] == [r.seed for r in b]         # reproducible (§10)
    assert {r.instruction for r in a}                          # instructions vary


def test_quality_subset_is_3x3x2():
    sub = quality_subset()
    assert len(sub) == 18                                      # 3 groups x 3 difficulty x 2 tasks
    assert {t.capability for t in sub} == {"pick_place", "articulated", "tool_use"}
    assert {t.difficulty for t in sub} == {"easy", "medium", "hard"}
    assert all(t.episodes == 10 for t in sub)


# -- §7 log schema ----------------------------------------------------------------
def test_jsonl_row_has_every_required_field():
    req = build_mock_replay(1)[0]
    rec = make_engine("mock", config_by_id("G3")).run_request(req)
    row = rec.to_jsonl_row(run_id="t", configuration="G3", engine="mock")
    assert set(row) >= {"run_id", "configuration", "engine", "task", "episode_id",
                        "request_id", "latency_ms", "denoising_step_ms", "peak_memory_mb",
                        "output_checksum", "quality_gate"}
    assert set(row["latency_ms"]) >= {"preprocess", "h2d", "reasoner", "generator_prepare",
                                      "denoising", "postprocess", "d2h", "server_total",
                                      "first_action", "chunk_total"}
    assert len(row["denoising_step_ms"]) == N_DENOISE_STEPS     # one timer per denoising step


def test_action_chunk_is_32x8():
    assert ACTION_CHUNK == (32, 8)


def test_generator_sampling_recipe():
    """The DROID generator recipe: steps=4, guidance=3, shift=5, full-range CFG Null.
    N_DENOISE_STEPS (and thus the denoising_step_ms length) derives from steps."""
    s = GENERATOR_SAMPLING
    assert (s.steps, s.guidance, s.shift, s.cfg_mode) == (4, 3.0, 5.0, "full-range-null")
    assert N_DENOISE_STEPS == s.steps == 4
    assert s.uses_cfg is True                                  # gates the CFG-Parallel job (§3)


def test_reasoner_sampling_is_deterministic():
    """Reasoner conditioning is decoded greedily (temperature 0) so logs are reproducible
    (§10). serving.reasoner_sampling_params() must surface the recipe."""
    from policy.serving import reasoner_sampling_params
    s = REASONER_SAMPLING
    assert s.temperature == 0.0 and s.greedy is True
    assert s.max_tokens > 0
    assert reasoner_sampling_params()["max_tokens"] == s.max_tokens


# -- Pipeline model invariants ----------------------------------------------------
def test_reasoner_cache_amortizes_conditioning():
    req = build_mock_replay(1)[0]
    r0 = make_engine("mock", config_by_id("R0")).run_request(req)   # recompute per step
    r3 = make_engine("mock", config_by_id("R3")).run_request(req)   # cache -> once
    # Naive recomputes conditioning every denoising step; caching does it once, so the win
    # scales with the step count (~N_DENOISE_STEPS x), not a fixed factor.
    assert r0.reasoner_ms > (N_DENOISE_STEPS - 1) * r3.reasoner_ms


def test_generator_waterfall_isolates_the_reasoner():
    # In the generator waterfall the reasoner is held at its single-conditioning cost, so it
    # is (nearly) constant across G0..G4 — only generator stages move.
    req = build_mock_replay(1)[0]
    reasoners = [make_engine("mock", c).run_request(req).reasoner_ms for c in GENERATOR_LADDER]
    assert max(reasoners) - min(reasoners) < 5.0                    # ~constant (only jitter)


def test_lossless_configs_are_bit_identical_to_baseline():
    # Lossless techniques must produce the same action checksum as eager (numerical
    # equivalence); lossy ones (Cache-DiT/FP8) deviate.
    req = build_mock_replay(1)[0]
    base = make_engine("mock", config_by_id("E0")).run_request(req).output_checksum
    e4 = make_engine("mock", config_by_id("E4")).run_request(req).output_checksum   # lossless
    e6 = make_engine("mock", config_by_id("E6")).run_request(req).output_checksum   # +fp8/cache-dit
    assert e4 == base
    assert e6 != base


def test_mock_is_reproducible_across_engine_instances():
    req = build_mock_replay(1)[0]
    a = make_engine("mock", config_by_id("E3")).run_request(req)
    b = make_engine("mock", config_by_id("E3")).run_request(req)
    assert a.total_chunk_ms == b.total_chunk_ms                     # no per-process nondeterminism


# -- End-to-end: matrix -> aggregate (§4, §8, §10) --------------------------------
@pytest.fixture(scope="module")
def aggregated(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    manifest_path = out / "replay/manifest.json"
    write_mock_manifest(manifest_path, n=40)              # stage the fixed replay set locally (§8)
    exp = Experiment(backend="mock", output_dir=str(out),
                     input_manifest=str(manifest_path), replay_size=40,
                     warmup_requests=5, wait_between_seconds=0.0)
    status = run_matrix(exp, spawn=False)
    manifest = aggregate(out, make_plots=False)
    return out, status, manifest


def test_matrix_runs_full_grid_and_baseline_drift_zero(aggregated):
    out, status, _ = aggregated
    assert not status["failed"]
    # every config plus the repeated end-baseline
    assert set(status["configurations_run"]) >= {c.cid for c in all_configs()} | {"E0_end"}
    assert status["baseline_drift"]["drift_pct"] == 0.0            # deterministic mock
    assert not status["rejected"]


def test_every_waterfall_is_monotonic_non_increasing(aggregated):
    out, _, _ = aggregated
    results = load_results(out)
    for wf in (REASONER, GENERATOR, END_TO_END):
        from policy.aggregate import build_waterfall
        rungs = build_waterfall(results, wf)["rungs"]
        p50 = [r["p50_ms"] for r in rungs]
        assert all(a >= b - 1e-6 for a, b in zip(p50, p50[1:])), wf   # never slower
        assert all(r["vs_v0"] >= 1.0 for r in rungs)


def test_stage_breakdown_reconciles_to_wall_clock(aggregated):
    out, _, _ = aggregated
    sb = build_stage_breakdown(load_results(out))
    # the six §3 stages are exactly the buckets, and the final is faster than baseline
    assert set(sb["baseline"]) == set(STAGES)
    assert sb["final_total_ms"] < sb["baseline_total_ms"]


def test_final_acceptance_checks(aggregated):
    _, _, manifest = aggregated
    acc = manifest["final_acceptance"]
    assert acc["lower_p50_chunk_latency"] and acc["lower_p99_chunk_latency"]
    assert acc["end_to_end_speedup_p50"] > 1.0
    assert acc["all_lossy_gates_passed"]


# -- RoboLab quality gate (§5, §9) ------------------------------------------------
def test_robolab_subset_gate_passes_for_final():
    result = robolab.compare("E0", "E6", backend="mock")
    assert result["success_drop"] <= result["threshold"]
    assert result["passed"]
    assert 0.0 <= result["candidate_success"] <= 1.0


# -- Native PyTorch backend / technique compatibility (§5.3.1 vs §5.3.3) -----------
def test_pytorch_backend_rejects_vllm_only_techniques():
    # Cache-DiT (G3/E5) and FP8 (G4/E6) are vLLM-Omni-only (§5.3.3) — not native PyTorch.
    for cid in ("G3", "G4", "E5", "E6"):
        c = config_by_id(cid)
        assert not compat.supported(c, "pytorch")
        with pytest.raises(UnsupportedTechnique):
            make_engine("pytorch", c)          # __init__ validates -> refuses
    # The §5.3.1 rungs ARE native-PyTorch (R0-R3, the lossless G0-G2, and E0-E4).
    for cid in ("R0", "R1", "R2", "R3", "G0", "G1", "G2", "E0", "E1", "E2", "E3", "E4"):
        assert compat.supported(config_by_id(cid), "pytorch")


def test_vllm_backend_supports_every_config():
    assert all(compat.supported(c, "vllm") for c in all_configs())


def test_cachedit_cudagraph_conflict_is_flagged():
    # E5/E6 stack CUDA graphs (from E3) + Cache-DiT (E5) — a non-composing pair (§9).
    assert compat.conflicts(config_by_id("E5"))
    assert compat.conflicts(config_by_id("E6"))
    assert not compat.conflicts(config_by_id("E4"))   # no Cache-DiT yet -> no conflict


def test_end_to_end_and_lossy_route_to_vllm():
    # A native-PyTorch waterfall run routes the production-stack configs to vLLM/vLLM-Omni:
    #   the whole end-to-end (E) ladder + the Cache-DiT/FP8 rungs (§5.3.2/§5.3.3).
    for cid in ("E0", "E1", "E2", "E3", "E4", "E5", "E6"):
        assert compat.resolve_backend(config_by_id(cid), "pytorch") == "vllm"
    for cid in ("G3", "G4"):                          # Cache-DiT / FP8 -> vLLM-Omni
        assert compat.resolve_backend(config_by_id(cid), "pytorch") == "vllm"
    for cid in ("R0", "R3", "G0", "G2"):              # native §5.3.1 reference rungs
        assert compat.resolve_backend(config_by_id(cid), "pytorch") == "pytorch"
    for c in all_configs():                           # mock dry-run: everything modeled
        assert compat.resolve_backend(c, "mock") == "mock"


def test_matrix_routes_configs_per_backend(tmp_path):
    # A pytorch run records which backend each config ran on (E + Cache-DiT/FP8 -> vllm).
    mp = tmp_path / "replay" / "manifest.json"
    write_mock_manifest(mp, n=4)
    exp = Experiment(backend="pytorch", output_dir=str(tmp_path), input_manifest=str(mp),
                     replay_size=2, warmup_requests=0, wait_between_seconds=0.0,
                     baseline_at_start_and_end=False, randomize_order=False)
    status = run_matrix(exp, spawn=False)             # everything fails off-box; routing still recorded
    cb = status["config_backends"]
    assert cb["E6"] == "vllm" and cb["G4"] == "vllm" and cb["R0"] == "pytorch"
    assert not status["skipped"]                      # nothing skipped — all routed to a real backend
