"""Single source of truth for every run parameter (config/experiment.yaml).

Loads config/experiment.yaml ONCE at import and exposes it as `CONFIG`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "experiment.yaml"


@dataclass(frozen=True)
class RunConfig:
    """Run-level knobs + bias controls (the record-of-record for a provisioned job)."""
    run_id: str = "cosmos-droid-001"
    model: str = "nvidia/Cosmos3-Nano-Policy-DROID"
    checkpoint_dir: str = "/local/model"          # weights staged locally before timing
    backend: str = "mock"                         # mock (no GPU) | vllm (target inference GPU)
    endpoint: str | None = None                   # vllm: externally-deployed endpoint (else launched)
    input_manifest: str = "policy/mock/manifest.json"
    output_dir: str = "results"
    torchinductor_root: str = "/tmp/torchinductor"   # per-config subdir /tmp/torchinductor/<cid>
    configurations: tuple = ()                    # cids to run; empty = full matrix (P0-P3,E0-E4)
    baseline_at_start_and_end: bool = True
    randomize_order: bool = True
    order_seed: int = 7
    wait_between_seconds: float = 5.0
    baseline_drift_reject_pct: float = 10.0       # reject if the two E0 baselines differ by more


@dataclass(frozen=True)
class DatasetConfig:
    """DROID observation shapes (static → CUDA-graph friendly) + RoboLab quality subset."""
    camera_views: tuple = ("exterior", "wrist")   # DROID convention: exterior + wrist
    image_hw: tuple = (180, 320)                  # per-view RGB fed to the reasoner (VERIFY on-box)
    proprio_dim: int = 8                          # joint pos/vel + gripper
    instruction_tokens: int = 32                  # tokenized language-instruction length (bucketed)
    action_chunk: tuple = (32, 8)                 # the only evaluated task: 32 steps × 8 DoF
    replay_size: int = 50                         # MEASURED requests/config = the 50 unique obs, once each
    replay_seed: int = 20260713                   # deterministic → reproducible logs
    capability_groups: tuple = ("visual", "relational", "procedural")   # RoboLab attribute families
    difficulty_levels: tuple = ("easy", "medium", "hard")
    tasks_per_cell: int = 2                        # 3 groups × 3 difficulty × 2 = 18 tasks
    episodes_per_task: int = 10


@dataclass(frozen=True)
class GeneratorSampling:
    """Generator (action-diffusion) sampling recipe for Cosmos3-Nano-Policy-DROID.

    Same recipe across the whole waterfall so techniques (not a changed schedule) explain
    latency deltas. `steps` is a static shape the CUDA-graph rungs capture."""
    steps: int = 4                      # denoising / flow-matching steps per action chunk
    guidance: float = 3.0              # classifier-free guidance scale
    shift: float = 5.0                 # flow-matching timestep-schedule shift
    cfg_mode: str = "full-range-null"  # CFG over the full step range against a null (uncond) branch

    @property
    def uses_cfg(self) -> bool:
        """CFG active -> the CFG-Parallel multi-GPU experiment applies."""
        return self.cfg_mode != "none" and self.guidance > 1.0


@dataclass(frozen=True)
class ReasonerSampling:
    """Reasoner (Qwen3-VL conditioning) decode params. Measured ONLY as conditioning, not
    standalone text; deterministic (temperature 0) for reproducibility."""
    max_tokens: int = 256          # conditioning-token budget per observation (VERIFY on-box)
    temperature: float = 0.0       # greedy / deterministic conditioning (reproducible)
    top_p: float = 1.0             # no nucleus truncation (inactive at temperature 0)
    top_k: int = -1                # disabled (vLLM convention)
    repetition_penalty: float = 1.0  # none (1.0 == off)

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass(frozen=True)
class MeasurementConfig:
    """Batch 1, ~50 warm-ups (excluded), measured = replay_size, p50/p90/p99 (p99 rough at n=50)."""
    warmup_requests: int = 50
    min_measured_requests: int = 50
    percentiles: tuple = (50, 90, 99)


@dataclass(frozen=True)
class QualityGateConfig:
    """Lossy-technique (Cache-DiT / FP8) accept/reject thresholds."""
    action_mse_threshold: float = 0.02   # mock action-drift gate (policy/pipeline.py)
    robolab_success_drop: float = 0.03   # RoboLab-subset success-drop gate (policy/robolab.py)


@dataclass(frozen=True)
class Settings:
    """Root of the single config file — one attribute per YAML section."""
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    reasoner_sampling: ReasonerSampling = field(default_factory=ReasonerSampling)
    generator_sampling: GeneratorSampling = field(default_factory=GeneratorSampling)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)


_SECTIONS: dict[str, type] = {
    "run": RunConfig,
    "dataset": DatasetConfig,
    "reasoner_sampling": ReasonerSampling,
    "generator_sampling": GeneratorSampling,
    "measurement": MeasurementConfig,
    "quality_gate": QualityGateConfig,
}
# Fields declared as tuples above — YAML gives lists, coerce so shapes stay hashable/static.
_TUPLE_FIELDS = {"configurations", "camera_views", "image_hw", "action_chunk",
                 "capability_groups", "difficulty_levels", "percentiles"}


def _build(cls: type, data: dict):
    """Merge one YAML section over a section dataclass's defaults, rejecting unknown keys."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}; known: {sorted(known)}")
    kw = {k: (tuple(v) if k in _TUPLE_FIELDS and isinstance(v, list) else v)
          for k, v in data.items()}
    return cls(**kw)


def load_config(path: str | Path | None = None) -> Settings:
    """Load the config file into `Settings` (code defaults fill absent sections/keys)."""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return Settings()
    data = yaml.safe_load(p.read_text()) or {}
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}; known: {sorted(_SECTIONS)}")
    return Settings(**{name: _build(cls, data.get(name) or {}) for name, cls in _SECTIONS.items()})


CONFIG: Settings = load_config()
