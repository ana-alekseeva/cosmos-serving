"""Minimal sanity tests for the optimization harness — `uv run pytest`.

Tests LOGIC and INVARIANTS, not the mock speedup numbers (illustrative, replaced by
real vLLM measurements on the H200) nor matplotlib rendering.
"""
from pathlib import Path

import pytest

from bench.ablation import run_ablation
from bench.plots import (
    plot_batching_throughput,
    plot_contribution_waterfall,
    plot_reasoner_sweep,
)
from bench.sweep import run_batching_sweep, run_reasoner_sweep
from bench.workload import BATCHING_TABLE9, ops_for
from optimize.registry import (
    GENERATOR,
    REASONER,
    ablation_ladder,
    resolve,
    techniques_for,
)


# -- Generator: technique selection + latency waterfall ---------------------------
def test_resolve_selection():
    assert resolve(GENERATOR, preset="none") == []
    # --enable returns ladder order, not argument order
    assert [t.key for t in resolve(GENERATOR, enable=["fp8", "cuda-graphs"])] == ["cuda-graphs", "fp8"]
    with pytest.raises(ValueError):
        resolve(GENERATOR, enable=["bogus"])


def test_reasoner_has_no_technique_ladder():
    # §5.3.2: the reasoner is stock vLLM out of the box — no toggles, no waterfall.
    assert techniques_for(REASONER) == []
    assert ablation_ladder(REASONER) == []


def test_generator_ablation_is_well_formed():
    # the waterfall data: baseline first, one variant per ladder technique, never slower
    res = run_ablation(GENERATOR, backend="mock")
    assert res.variants[0].technique is None
    assert len(res.variants) == len(ablation_ladder(GENERATOR)) + 1
    for op in res.ops:
        lat = res.latencies(op.name)
        assert all(a >= b - 1e-6 for a, b in zip(lat, lat[1:]))          # never gets slower
        assert all(r["vs_v0"] >= 1.0 for r in res.marginal_rows(op.name))


def test_every_generator_op_has_speedups():
    # guard the #1 silent bug: a new OP with no speedup key -> mock returns 1.0
    op_names = {op.name for op in ops_for(GENERATOR)}
    for t in techniques_for(GENERATOR):
        missing = op_names - set(t.mock_speedups)
        assert not missing, f"{t.key} missing mock_speedups for {missing}"


def test_cuda_graphs_reproduces_t2i_headline():
    # report §5.3.1: "CUDA Graphs on T2I yielded 30% to 60% speedups". The marginal drop
    # of the cuda-graphs rung on the T2I panel must land in that band.
    res = run_ablation(GENERATOR, backend="mock")
    rows = res.marginal_rows("t2i-1024")
    cg = next(r for r in rows if "CUDA graphs" in r["variant"])
    drop = 100.0 * (1.0 - 1.0 / cg["vs_prev"])
    assert 30.0 <= drop <= 60.0, f"T2I CUDA-graph drop {drop:.0f}% outside the reported 30-60%"


def test_scaling_and_memory_off_the_latency_ladder():
    ladder = {t.key for t in ablation_ladder(GENERATOR)}
    allt = {t.key for t in techniques_for(GENERATOR)}
    assert [t.key for t in techniques_for(GENERATOR) if t.scaling] == ["cfg-parallel", "context-parallel"]
    assert {"hsdp", "cpu-offload", "batching"} <= allt          # selectable via --enable
    assert not ({"hsdp", "cpu-offload", "batching"} & ladder)   # excluded from the latency waterfall


def test_distributed_group_is_mutually_exclusive():
    full = [t.key for t in resolve(GENERATOR, preset="full")]
    assert "cfg-parallel" in full and "context-parallel" not in full
    with pytest.raises(ValueError):
        resolve(GENERATOR, enable=["cfg-parallel", "context-parallel"])


# -- Reasoner: concurrency/shape sweep --------------------------------------------
def test_reasoner_sweep_monotonic_throughput():
    res = run_reasoner_sweep(backend="mock")
    for shape in res.shapes:
        tput = [y for _, y in res.curve(shape, "throughput_tok_s")]
        ttft = [y for _, y in res.curve(shape, "ttft_ms")]
        assert tput == sorted(tput)        # throughput rises with concurrency
        assert ttft == sorted(ttft)        # TTFT rises (queueing) with concurrency
        assert all(t > 0 for t in ttft)


# -- Generator: batching throughput (Table 9) -------------------------------------
def test_batching_sweep_reproduces_table9():
    res = run_batching_sweep(backend="mock")
    got = {(r["resolution"], r["series"]): r["gain_pct"] for r in res.rows}
    for resn, series in BATCHING_TABLE9.items():
        for s, pct in series.items():
            assert got[(resn, s)] == float(pct)


# -- Plots render -----------------------------------------------------------------
def test_plots_render(tmp_path: Path):
    assert plot_contribution_waterfall(run_ablation(GENERATOR, backend="mock"),
                                       tmp_path / "wf.png").stat().st_size > 0
    assert plot_reasoner_sweep(run_reasoner_sweep(backend="mock"),
                               tmp_path / "sweep.png").stat().st_size > 0
    assert plot_batching_throughput(run_batching_sweep(backend="mock"),
                                    tmp_path / "batch.png").stat().st_size > 0
