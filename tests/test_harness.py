"""Minimal sanity tests for the optimization harness — `uv run pytest`.

Tests LOGIC and INVARIANTS, not the mock speedup numbers (illustrative, replaced
by real vLLM measurements on the H200) nor matplotlib rendering.
"""
from pathlib import Path

import pytest

from bench.ablation import run_ablation
from bench.plots import plot_contribution_waterfall
from bench.workload import ops_for
from optimize.registry import (
    GENERATOR,
    REASONER,
    ablation_ladder,
    resolve,
    techniques_for,
)


def test_resolve_selection():
    # the `optimize` command's core: presets + subset selection
    assert resolve(REASONER, preset="none") == []
    assert resolve(REASONER, preset="full") == techniques_for(REASONER)  # no groups -> all
    # --enable returns ladder order, not argument order
    assert [t.key for t in resolve(REASONER, enable=["fp8", "cuda-graphs"])] == ["cuda-graphs", "fp8"]
    with pytest.raises(ValueError):
        resolve(REASONER, enable=["bogus"])


@pytest.mark.parametrize("tower", [REASONER, GENERATOR])
def test_ablation_is_well_formed(tower):
    # the waterfall data: baseline first, one variant per ladder technique, never slower
    res = run_ablation(tower, backend="mock")
    assert res.variants[0].technique is None
    assert len(res.variants) == len(ablation_ladder(tower)) + 1
    for op in res.ops:
        lat = res.latencies(op.name)
        assert all(a >= b - 1e-6 for a, b in zip(lat, lat[1:]))          # never gets slower
        assert all(r["vs_v0"] >= 1.0 for r in res.marginal_rows(op.name))


def test_every_op_has_speedups():
    # guard the #1 silent bug: a new OP with no speedup key -> mock returns 1.0
    for tower in (REASONER, GENERATOR):
        op_names = {op.name for op in ops_for(tower)}
        for t in techniques_for(tower):
            missing = op_names - set(t.mock_speedups)
            assert not missing, f"{t.key} missing mock_speedups for {missing}"


def test_scaling_flag_marks_distributed_techniques():
    # plot color/legend + the 2-GPU honesty caveat depend on this flag
    assert [t.key for t in techniques_for(GENERATOR) if t.scaling] == ["cfg-parallel", "context-parallel"]


def test_distributed_group_is_mutually_exclusive():
    # full preset collapses the group to one strategy; enabling both raises
    full = [t.key for t in resolve(GENERATOR, preset="full")]
    assert "cfg-parallel" in full and "context-parallel" not in full
    with pytest.raises(ValueError):
        resolve(GENERATOR, enable=["cfg-parallel", "context-parallel"])


def test_memory_techniques_selectable_but_off_the_waterfall():
    ladder = {t.key for t in ablation_ladder(GENERATOR)}
    allt = {t.key for t in techniques_for(GENERATOR)}
    assert {"hsdp", "cpu-offload"} <= allt          # selectable via --enable / in full
    assert not ({"hsdp", "cpu-offload"} & ladder)   # excluded from the latency waterfall


def test_waterfall_png_written(tmp_path: Path):
    # smoke: the primary artifact actually renders
    out = plot_contribution_waterfall(run_ablation(REASONER, backend="mock"), tmp_path / "wf.png")
    assert out.exists() and out.stat().st_size > 0
