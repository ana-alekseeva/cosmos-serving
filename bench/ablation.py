"""Cumulative ablation runner — the `--ablate` engine (specification.md §4, §6, §9).

Walks the canonical technique ladder, adding one technique at a time, measures each
variant across every operating point, and produces the "vs V0 / vs prev" table that
becomes the contribution waterfall. Mirrors gpu_and_inference_hw/hw2/ablation.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from bench.drivers import make_engine
from bench.workload import OperatingPoint, ops_for
from optimize.registry import REASONER, Technique, ablation_ladder


@dataclass(frozen=True)
class Variant:
    index: int                       # 0 = baseline, i = after adding ladder[i-1]
    label: str                       # "R0 baseline" / "+ KV cache"
    technique: Technique | None      # None for the baseline
    enabled: tuple[Technique, ...]   # cumulative set active at this variant


@dataclass
class AblationResult:
    tower: str
    backend: str
    ops: list[OperatingPoint]
    variants: list[Variant]
    p50: dict = field(default_factory=dict)   # (variant_index, op_name) -> ms
    p95: dict = field(default_factory=dict)

    def latencies(self, op_name: str) -> list[float]:
        return [self.p50[(v.index, op_name)] for v in self.variants]

    def marginal_rows(self, op_name: str) -> list[dict]:
        """Per-variant rows with cumulative (vs V0) and marginal (vs prev) speedup."""
        rows, base, prev = [], None, None
        for v in self.variants:
            ms = self.p50[(v.index, op_name)]
            base = ms if base is None else base
            vs0 = base / ms if ms else float("inf")
            vsp = (prev / ms) if (prev and ms) else 1.0
            rows.append({
                "variant": v.label,
                "ms": round(ms, 1),
                "vs_v0": round(vs0, 2),
                "vs_prev": round(vsp, 2),
                "lossy": bool(v.technique and v.technique.lossy),
                "scaling": bool(v.technique and v.technique.scaling),
            })
            prev = ms
        return rows

    def to_dict(self) -> dict:
        return {
            "tower": self.tower,
            "backend": self.backend,
            "ops": [op.name for op in self.ops],
            "variants": [v.label for v in self.variants],
            "results": {
                op.name: self.marginal_rows(op.name) for op in self.ops
            },
        }


def _baseline_label(tower: str) -> str:
    return ("R0" if tower == REASONER else "G0") + " baseline"


def run_ablation(tower: str, *, backend: str = "mock",
                 ops: list[OperatingPoint] | None = None,
                 repeats: int = 10) -> AblationResult:
    ladder = ablation_ladder(tower)
    ops = ops or ops_for(tower)

    variants = [Variant(0, _baseline_label(tower), None, ())]
    for i, tech in enumerate(ladder, start=1):
        variants.append(Variant(i, f"+ {tech.label}", tech, tuple(ladder[:i])))

    result = AblationResult(tower=tower, backend=backend, ops=ops, variants=variants)
    for v in variants:
        engine = make_engine(backend, list(v.enabled))
        for op in ops:
            m = engine.measure(op, repeats=repeats)
            result.p50[(v.index, op.name)] = m.p50_ms
            result.p95[(v.index, op.name)] = m.p95_ms
    return result


def print_summary(result: AblationResult) -> None:
    """Console table per operating point (à la hw2 ablation SUMMARY)."""
    for op in result.ops:
        print("\n" + "=" * 72)
        print(f"[{result.tower}] OP {op.label()}   (backend={result.backend})")
        print("=" * 72)
        print(f"{'Variant':<34}{'ms':>12}{'vs V0':>10}{'vs prev':>10}")
        print("-" * 72)
        for row in result.marginal_rows(op.name):
            tag = "  *2-GPU" if row["scaling"] else ("  *lossy" if row["lossy"] else "")
            print(f"{row['variant']:<34}{row['ms']:>12.1f}"
                  f"{row['vs_v0']:>9.2f}x{row['vs_prev']:>9.2f}x{tag}")
        print("-" * 72)
    print("\n'vs prev' is the marginal speedup that one technique added.")


if __name__ == "__main__":  # stdlib-only entry: `python -m bench.ablation [tower]`
    import sys
    tower = sys.argv[1] if len(sys.argv) > 1 else REASONER
    print_summary(run_ablation(tower))
