"""Per-configuration log artifacts.

Each subprocess (run_configuration.py) writes, into its own output directory:

    <cid>.jsonl        one row per request (the minimal log format)
    summary.json       p50/p90/p99 rollups + config metadata
    environment.json   package versions / engine flags (reproducibility)
    system-info.json   GPU model, driver, clocks, temperature (drift accounting)
    status.json        started/finished/ok, warmup + measured counts, baseline flag

The aggregation job merges every subprocess's files.
"""
from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from policy.configs import GENERATOR_SAMPLING, REASONER_SAMPLING
from policy.measure import LatencyRecord, summarize


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(records: list[LatencyRecord], path: str | Path, *,
                run_id: str, configuration: str, engine: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r.to_jsonl_row(run_id=run_id, configuration=configuration,
                                               engine=engine)) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_summary(records: list[LatencyRecord], path: str | Path, *,
                  run_id: str, configuration: str, engine: str, waterfall: str,
                  lossy: bool, warmups: int, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "configuration": configuration,
        "waterfall": waterfall,
        "engine": engine,
        "batch_size": 1,               # batch size 1 for all latency measurements
        "lossy_quality_gated": lossy,
        "warmup_requests": warmups,
        "measured_requests": len(records),
        "percentiles": summarize(records),
        **(extra or {}),
    }
    p.write_text(json.dumps(payload, indent=2))
    return p


def write_environment(path: str | Path, *, engine: str, model: str,
                      stage_flags: dict, torchinductor_dir: str | None = None) -> Path:
    """Package versions + engine flags. Best-effort import of torch/vllm (may be absent)."""
    def _ver(mod: str) -> str:
        try:
            return __import__(mod).__version__
        except Exception:
            return "absent"

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "captured_at": _now(),
        "engine": engine,
        "model": model,
        "python": sys.version.split()[0],
        "packages": {m: _ver(m) for m in ("numpy", "torch", "vllm", "matplotlib")},
        "stage_flags": stage_flags,
        "reasoner_sampling": asdict(REASONER_SAMPLING),     # max_tokens/temperature/...
        "generator_sampling": asdict(GENERATOR_SAMPLING),   # steps/guidance/shift/cfg
        "torchinductor_cache_dir": torchinductor_dir,
    }, indent=2))
    return p


def write_system_info(path: str | Path) -> Path:
    """GPU model / driver / clocks / temperature (drift accounting). Falls back to
    platform info when nvidia-smi / torch.cuda are unavailable (mock/CPU runs)."""
    info = {
        "captured_at": _now(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "gpu": _gpu_info(),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, indent=2))
    return p


def _gpu_info() -> dict:
    """nvidia-smi one-shot (name, driver, temperature, clocks) — empty on non-GPU hosts."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found (mock/CPU host)"}
    query = "name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except Exception as exc:                       # nvidia-smi present but failed
        return {"available": False, "reason": str(exc)[:200]}
    gpus = []
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        keys = ["name", "driver_version", "temperature_gpu_c", "clock_sm_mhz",
                "clock_mem_mhz", "power_draw_w"]
        gpus.append(dict(zip(keys, cols)))
    return {"available": True, "gpus": gpus}


def write_status(path: str | Path, *, configuration: str, ok: bool,
                 started_at: str, warmups: int, measured: int,
                 is_baseline: bool = False, error: str | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "configuration": configuration,
        "ok": ok,
        "is_baseline": is_baseline,
        "started_at": started_at,
        "finished_at": _now(),
        "warmup_requests": warmups,
        "measured_requests": measured,
        "error": error,
    }, indent=2))
    return p
