"""Per-stage timers for the stage-breakdown plot (specification.md §4, F5).

StageTimer accumulates wall-clock per named stage using CUDA events when a GPU is
present, else perf_counter. `reconcile()` checks the summed stages against the
black-box end-to-end wall-clock (acceptance: <=5%).
"""
from __future__ import annotations

import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self.stages_ms: dict[str, float] = {}
        self._cuda = self._cuda_available()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @contextmanager
    def stage(self, name: str):
        if self._cuda:
            import torch
            start, end = (torch.cuda.Event(enable_timing=True),
                          torch.cuda.Event(enable_timing=True))
            start.record()
            yield
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)  # ms
        else:
            t0 = time.perf_counter()
            yield
            elapsed = (time.perf_counter() - t0) * 1e3
        self.stages_ms[name] = self.stages_ms.get(name, 0.0) + elapsed

    def total_ms(self) -> float:
        return sum(self.stages_ms.values())

    def reconcile(self, end_to_end_ms: float, tol: float = 0.05) -> dict:
        summed = self.total_ms()
        rel = abs(summed - end_to_end_ms) / end_to_end_ms if end_to_end_ms else float("inf")
        return {
            "summed_ms": round(summed, 3),
            "end_to_end_ms": round(end_to_end_ms, 3),
            "rel_error": round(rel, 4),
            "ok": rel <= tol,
        }
