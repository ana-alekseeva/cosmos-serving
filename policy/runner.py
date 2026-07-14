"""Run ONE configuration end-to-end (specification_revised.txt §4 Job 1 subprocess).

Each configuration runs as its own subprocess (run_configuration.py) so its CUDA context
is released before the next one starts (§4). The subprocess:

    1. loads the model from local storage
    2. runs warm-up requests (excluded from timing, §6)
    3. runs the fixed latency replay set (the 50 unique obs, once each; §6)
    4. writes <cid>.jsonl + summary.json + environment.json + system-info.json + status.json
    5. exits (releasing the CUDA context)

This module is the importable core; run_configuration.py is the thin CLI wrapper.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from policy.configs import Config, config_by_id
from policy.dataset import DroidRequest, load_manifest, tile_to
from policy.experiment import Experiment
from policy.logs import (
    write_environment,
    write_jsonl,
    write_status,
    write_summary,
    write_system_info,
)
from policy.measure import LatencyRecord
from policy.pipeline import make_engine


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_torchinductor_cache(config: Config, root: str) -> str:
    """Per-configuration compilation-cache directory (§4): /tmp/torchinductor/<cid>.
    Isolates each config's torch.compile cache so timings don't cross-contaminate."""
    cache_dir = str(Path(root) / config.cid)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def run_configuration(
    config: Config,
    requests: list[DroidRequest],
    *,
    backend: str = "mock",
    out_dir: str | Path = "results",
    run_id: str = "cosmos-droid-001",
    model: str | None = None,
    endpoint: str | None = None,
    warmups: int = 25,
    is_baseline: bool = False,
    torchinductor_root: str = "/tmp/torchinductor",
    out_subdir: str | None = None,
) -> dict:
    """Measure `config` over `requests`; write the five §7 artifacts into out_dir/<subdir>/.

    `out_subdir` defaults to the config id; the matrix passes e.g. "E0_end" for the repeated
    end-of-run baseline so it does not overwrite the start baseline (§8)."""
    cfg_dir = Path(out_dir) / (out_subdir or config.cid)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    started = _now()
    cache_dir = set_torchinductor_cache(config, torchinductor_root)

    engine = make_engine(backend, config, model=model, endpoint=endpoint)
    records: list[LatencyRecord] = []
    error = None
    try:
        engine.prepare()                                # load model / launch server
        for req in requests[:warmups]:                  # warm-up (compile, caches) — discarded
            engine.run_request(req)
        for req in requests:                            # measured pass (fixed replay set)
            records.append(engine.run_request(req))
    except Exception as exc:                            # a failed config must not kill the matrix
        error = f"{type(exc).__name__}: {exc}"[:1500]
    finally:
        engine.close()

    write_jsonl(records, cfg_dir / f"{config.cid}.jsonl",
                run_id=run_id, configuration=config.cid, engine=backend)
    extra = {"quality_drift": _quality_drift(config, backend)}
    write_summary(records, cfg_dir / "summary.json", run_id=run_id, configuration=config.cid,
                  engine=backend, waterfall=config.waterfall, lossy=config.lossy,
                  warmups=warmups, extra=extra)
    write_environment(cfg_dir / "environment.json", engine=backend, model=model or "",
                      stage_flags=config.stage_flags, torchinductor_dir=cache_dir)
    write_system_info(cfg_dir / "system-info.json")
    write_status(cfg_dir / "status.json", configuration=config.cid, ok=error is None,
                 started_at=started, warmups=warmups, measured=len(records),
                 is_baseline=is_baseline, error=error)
    if error:
        raise RuntimeError(f"configuration {config.cid} failed: {error}")
    return {"configuration": config.cid, "measured": len(records), "dir": str(cfg_dir)}


def _quality_drift(config: Config, backend: str) -> float | None:
    """Modeled action drift for lossy configs (mock only) — the RoboLab subset is the real
    gate (§5/§9). Lossless configs are exact-match (drift 0)."""
    if backend != "mock":
        return None
    from policy.mock.engine import _quality_gate
    _, drift = _quality_gate(config)
    return drift


def resolve_configs(cids: list[str] | None):
    """cids -> Config objects (or the full matrix if empty/None)."""
    from policy.configs import all_configs
    if not cids:
        return all_configs()
    return [config_by_id(c) for c in cids]


def load_requests(exp: Experiment) -> list[DroidRequest]:
    """Load the replay manifest (§5), sized to replay_size measured requests.

    Default replay_size == the unique set, so every observation is measured once (tile_to is a
    pass-through); a smoke run (replay_size=1) takes the first one, and a larger replay_size
    cycles the set for tighter tails. A missing manifest raises (regenerate the mock fixture
    with `python -m policy.mock.replay` or stage a real capture manifest)."""
    return tile_to(load_manifest(exp.input_manifest), exp.replay_size)
