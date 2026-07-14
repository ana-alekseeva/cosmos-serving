"""Run-facing view of the single config file (config/experiment.yaml).

`policy/config.py` is the single source of truth (sectioned: run / dataset /
generator_sampling / measurement / quality_gate). This module projects the run-relevant
values into a FLAT `Experiment` — the shape run_matrix.py / policy.matrix / policy.runner
already consume (exp.replay_size, exp.warmup_requests, ...) — and applies CLI overrides.

CLI flags override the file; the file overrides code defaults; with no YAML at all the code
defaults (in policy/config.py) still run the harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from policy.config import CONFIG, Settings, load_config

_R, _D, _M = CONFIG.run, CONFIG.dataset, CONFIG.measurement


@dataclass
class Experiment:
    """Flat run view. Defaults mirror the loaded config so `Experiment()` == the YAML."""
    run_id: str = _R.run_id
    model: str = _R.model
    checkpoint_dir: str = _R.checkpoint_dir
    backend: str = _R.backend                    # mock | vllm
    endpoint: str | None = _R.endpoint           # vllm: externally-deployed endpoint URL

    input_manifest: str = _R.input_manifest
    replay_size: int = _D.replay_size            # dataset section (the fixed replay set, §5)
    replay_seed: int = _D.replay_seed

    output_dir: str = _R.output_dir
    torchinductor_root: str = _R.torchinductor_root   # per-config subdir (§4)

    warmup_requests: int = _M.warmup_requests         # measurement section (§6)
    min_measured_requests: int = _M.min_measured_requests

    # configurations to run (cids). Empty -> the full matrix (P0-P3, E0-E6).
    configurations: list = field(default_factory=lambda: list(_R.configurations))

    # §8 bias controls for the single long job.
    baseline_at_start_and_end: bool = _R.baseline_at_start_and_end
    randomize_order: bool = _R.randomize_order
    order_seed: int = _R.order_seed
    wait_between_seconds: float = _R.wait_between_seconds
    baseline_drift_reject_pct: float = _R.baseline_drift_reject_pct   # reject if baselines differ by more

    def override(self, **kw) -> "Experiment":
        return replace(self, **{k: v for k, v in kw.items() if v is not None})


def experiment_from_settings(cfg: Settings) -> Experiment:
    """Flatten the sectioned config into the run-facing Experiment."""
    r, d, m = cfg.run, cfg.dataset, cfg.measurement
    return Experiment(
        run_id=r.run_id, model=r.model, checkpoint_dir=r.checkpoint_dir, backend=r.backend,
        endpoint=r.endpoint, input_manifest=r.input_manifest,
        replay_size=d.replay_size, replay_seed=d.replay_seed,
        output_dir=r.output_dir, torchinductor_root=r.torchinductor_root,
        warmup_requests=m.warmup_requests, min_measured_requests=m.min_measured_requests,
        configurations=list(r.configurations),
        baseline_at_start_and_end=r.baseline_at_start_and_end,
        randomize_order=r.randomize_order, order_seed=r.order_seed,
        wait_between_seconds=r.wait_between_seconds,
        baseline_drift_reject_pct=r.baseline_drift_reject_pct,
    )


def load_experiment(path: str | Path | None) -> Experiment:
    """Load the single config file (or code defaults) and project it into an Experiment."""
    return experiment_from_settings(load_config(path))
