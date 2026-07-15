"""Sanity tests for the Cosmos3-Nano-Policy-DROID harness — `uv run pytest`.

Tests LOGIC and INVARIANTS (waterfall monotonicity, the §7 log schema, reproducibility,
the §8 drift check, the §10 acceptance checks, the quality gate), not the mock's modeled
constants (those are replaced by real vLLM/vLLM-Omni measurements on the GPU) nor the
matplotlib rendering.
"""
import json
from dataclasses import replace as dc_replace
from pathlib import Path

import pytest

from policy.aggregate import aggregate, build_stage_breakdown, load_results
from policy.configs import (
    ACTION_CHUNK,
    END_TO_END,
    END_TO_END_LADDER,
    GENERATOR_SAMPLING,
    N_DENOISE_STEPS,
    NATIVE,
    NATIVE_LADDER,
    REASONER_SAMPLING,
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
from policy import robolab, robolab_runner


# -- Config matrix (§3) -----------------------------------------------------------
def test_ladders_match_spec():
    assert [c.cid for c in NATIVE_LADDER] == ["P0", "P1", "P2", "P3"]
    assert [c.cid for c in END_TO_END_LADDER] == ["E0", "E1", "E2", "E3", "E4"]
    # baselines add no technique (P starts from the cuDNN fused-attention baseline)
    for lad in (NATIVE_LADDER, END_TO_END_LADDER):
        assert lad[0].added == "" and not lad[0].lossy


def test_only_fp8_is_lossy():
    lossy = {c.cid for c in all_configs() if c.lossy}
    # FP8 (E4) is the only lossy rung left: Cache-DiT was removed from the ladder
    # (never activates on the 4-step schedule + bypasses the compiled transformer) and
    # the reasoner conditioning cache moved to patch-proposal status (not implementable
    # stock) — see the rationale in policy/configs.py. The native P ladder has no lossy rungs.
    assert lossy == {"E4"}


def test_end_to_end_ladder_is_the_union_spec_lists():
    added = [c.added for c in END_TO_END_LADDER[1:]]
    assert added == ["flash-attention", "torch.compile", "cuda-graphs", "fp8"]


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
    assert {t.capability for t in sub} == {"visual", "relational", "procedural"}
    assert {t.difficulty for t in sub} == {"easy", "medium", "hard"}
    assert all(t.episodes == 10 for t in sub)


# -- §7 log schema ----------------------------------------------------------------
def test_jsonl_row_has_every_required_field():
    req = build_mock_replay(1)[0]
    rec = make_engine("mock", config_by_id("E4")).run_request(req)
    row = rec.to_jsonl_row(run_id="t", configuration="E4", engine="mock")
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
    p0 = make_engine("mock", config_by_id("P0")).run_request(req)   # recompute per step
    p3 = make_engine("mock", config_by_id("P3")).run_request(req)   # cache -> once
    # Naive recomputes conditioning every denoising step; caching does it once, so the win
    # scales with the step count (~N_DENOISE_STEPS x), not a fixed factor.
    assert p0.reasoner_ms > (N_DENOISE_STEPS - 1) * p3.reasoner_ms


def test_native_ladder_reasoner_is_monotonic_and_cache_dominates():
    # Across the merged native ladder reasoner_ms only goes down (compile/graphs help a little,
    # the P3 conditioning cache a lot — it removes the per-step recompute).
    req = build_mock_replay(1)[0]
    rms = [make_engine("mock", config_by_id(c)).run_request(req).reasoner_ms
           for c in ("P0", "P1", "P2", "P3")]
    assert all(a >= b - 1e-6 for a, b in zip(rms, rms[1:]))         # non-increasing
    assert rms[3] < rms[2] / 2                                      # the cache is the big drop


def test_lossless_configs_are_bit_identical_to_baseline():
    # Lossless techniques must produce the same action checksum as eager (numerical
    # equivalence); the lossy one (FP8) deviates.
    req = build_mock_replay(1)[0]
    base = make_engine("mock", config_by_id("E0")).run_request(req).output_checksum
    e3 = make_engine("mock", config_by_id("E3")).run_request(req).output_checksum   # lossless top
    e4 = make_engine("mock", config_by_id("E4")).run_request(req).output_checksum   # +fp8 (lossy)
    assert e3 == base
    assert e4 != base


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
    for wf in (NATIVE, END_TO_END):
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
    result = robolab.compare("E0", END_TO_END_LADDER[-1].cid, backend="mock")
    assert result["success_drop"] <= result["threshold"]
    assert result["passed"]
    assert 0.0 <= result["candidate_success"] <= 1.0


# -- Real RoboLab driver (§4 Job 3): everything testable without an Isaac box ------
def test_openpi_uri_forms():
    f = robolab_runner.openpi_uri
    route = robolab_runner.OPENPI_ROUTE
    assert f("https://ep.nebius.cloud") == f"wss://ep.nebius.cloud{route}"
    assert f("http://127.0.0.1:8000") == f"ws://127.0.0.1:8000{route}"
    assert f("https://host/base/") == f"wss://host/base{route}"
    assert f(f"wss://host{route}") == f"wss://host{route}"      # already the full URI


def test_committed_task_map_covers_every_slot():
    # The committed map fills all 18 slots with real RoboLab benchmark class names
    # (selection rationale in config/robolab_tasks.yaml; names validated against the
    # catalog's task_metadata.json at selection time).
    m = robolab_runner.load_task_map()
    assert len(m) == 18
    assert set(m) == {t.task for t in quality_subset()}
    assert all(v.endswith("Task") for v in m.values())


def test_task_map_fails_loudly_on_unfilled_slots(tmp_path):
    # An unfilled slot must refuse to run and be named — no invented task names.
    p = tmp_path / "map.yaml"
    p.write_text("RoboLab-visual-easy-0: null\n")
    with pytest.raises(ValueError, match="RoboLab-visual-easy-0"):
        robolab_runner.load_task_map(p)


def test_parse_task_success_accepts_plausible_shapes(tmp_path):
    cases = [
        ({"success_rate": 0.7, "num_runs": 10}, (0.7, 10)),
        ({"successes": 6, "episodes": 10}, (0.6, 10)),
        ({"results": [{"success": True}, {"success": False}]}, (0.5, 2)),
        ([True, True, False, False], (0.5, 4)),
    ]
    for i, (payload, want) in enumerate(cases):
        d = tmp_path / f"case{i}"
        d.mkdir()
        (d / "eval_summary.json").write_text(json.dumps(payload))
        assert robolab_runner.parse_task_success(d) == want
    # Name hints outrank alphabetical order: a metrics-free config.json must not win.
    d = tmp_path / "hinted"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"task": "X", "seed": 0}))
    (d / "success_metrics.json").write_text(json.dumps({"success_rate": 0.9}))
    assert robolab_runner.parse_task_success(d)[0] == 0.9
    # No parsable metrics -> loud failure listing what was seen, never a guessed number.
    d = tmp_path / "junk"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"task": "X"}))
    with pytest.raises(RuntimeError, match="config.json"):
        robolab_runner.parse_task_success(d)


def test_real_subset_resumes_from_records_and_matches_gate_shape(tmp_path):
    # Pre-written per-task records short-circuit the simulator entirely (§10 resume), so
    # this exercises the full real-path aggregation on a laptop. Shape must match the
    # mock's so compare() gates either backend.
    subset = quality_subset()
    cfg = END_TO_END_LADDER[-1]                       # the final optimized rung
    rollout = tmp_path / "robolab"
    for t in subset:
        p = rollout / cfg.cid / f"{t.task}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"task": t.task, "robolab_task": "MappedTask",
                                 "capability": t.capability, "difficulty": t.difficulty,
                                 "episodes": t.episodes, "success_rate": 0.8,
                                 "source": "test"}))
    result = robolab_runner.run_quality_subset_real(
        cfg, "https://ep.example", subset, robolab_root=tmp_path / "no-checkout-needed",
        rollout_dir=rollout, task_map={t.task: "MappedTask" for t in subset})
    assert result["overall_success"] == 0.8
    assert len(result["per_task"]) == len(subset)
    mock = robolab.run_quality_subset(cfg, backend="mock")
    assert set(mock["per_task"][0]) == set(result["per_task"][0])
    for key in ("configuration", "overall_success", "per_task"):
        assert key in result and key in mock


# -- Native PyTorch backend / technique compatibility (§5.3.1 vs §5.3.3) -----------
def test_pytorch_backend_rejects_vllm_only_techniques():
    # FP8 (E4) is vLLM-Omni-only (§5.3.3) — not native PyTorch.
    for cid in ("E4",):
        c = config_by_id(cid)
        assert not compat.supported(c, "pytorch")
        with pytest.raises(UnsupportedTechnique):
            make_engine("pytorch", c)          # __init__ validates -> refuses
    # The §5.3.1 rungs ARE native-PyTorch (the whole P ladder, and E0-E4).
    for cid in ("P0", "P1", "P2", "P3", "E0", "E1", "E2", "E3"):
        assert compat.supported(config_by_id(cid), "pytorch")


def test_vllm_backend_supports_every_config():
    assert all(compat.supported(c, "vllm") for c in all_configs())


def test_cachedit_cudagraph_conflict_is_flagged():
    # Cache-DiT + CUDA graphs is a non-composing pair (§9). No ladder rung carries
    # Cache-DiT anymore (removed after the Job 2 measurements), so the detector is
    # exercised on a hand-built off-ladder config; the ladder itself must be clean.
    e3 = config_by_id("E3")                            # cuda_graphs on
    off_ladder = dc_replace(e3, stage_flags={**e3.stage_flags, "cache_dit": True})
    assert compat.conflicts(off_ladder)
    assert all(not compat.conflicts(c) for c in all_configs())


def test_end_to_end_and_lossy_route_to_vllm():
    # A native-PyTorch waterfall run routes the production-stack configs to vLLM/vLLM-Omni:
    #   the whole end-to-end (E) ladder + the Cache-DiT/FP8 rungs (§5.3.2/§5.3.3).
    for cid in ("E0", "E1", "E2", "E3", "E4"):
        assert compat.resolve_backend(config_by_id(cid), "pytorch") == "vllm"
    for cid in ("P0", "P1", "P2", "P3"):              # native §5.3.1 reference rungs
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
    assert cb["E4"] == "vllm" and cb["P0"] == "pytorch"
    assert not status["skipped"]                      # nothing skipped — all routed to a real backend
