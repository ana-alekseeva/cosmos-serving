"""Single source of truth for every run parameter (config/experiment.yaml).

ALL parameters that govern a run live in one file — `config/experiment.yaml` — so you can
read what a run used without opening each module. This module loads that YAML ONCE at
import and exposes it as `CONFIG`; every other module pulls its numbers from here instead
of hard-coding them, so the file and the code cannot drift.

Sections (see config/experiment.yaml):
    run                — run id, model, backend, endpoint, paths, §8 bias controls
    dataset            — DROID observation shapes + RoboLab quality-subset structure (§5)
    generator_sampling — action-diffusion recipe: steps / guidance / shift / CFG mode
    measurement        — warm-ups, min measured, percentiles (§6)
    quality_gate       — lossy-technique accept/reject thresholds (§9)

What is deliberately NOT here (per design): the mock simulator's internal anchors — the
per-stage cost table and per-technique speedup multipliers (policy/pipeline.py,
policy/configs.py, policy/multigpu.py). Those are the *model's* placeholder numbers, thrown
away the moment the real vLLM / vLLM-Omni backend measures wall-clock on the GPU; they are
not "parameters of a run". Everything a human would call a hyperparameter is here.

NB: distinct from policy/configs.py, which BUILDS the optimization ladders (P0-P3,
E0-E6) from those simulator anchors. This module only holds declarative parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

# The one config file. Shipped in-repo as the canonical record; load_config(path) can point
# at another copy (run_matrix.py --config ...).
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "experiment.yaml"


@dataclass(frozen=True)
class RunConfig:
    """Run-level knobs + §8 bias controls (the record-of-record for a provisioned job)."""
    run_id: str = "cosmos-droid-001"
    model: str = "nvidia/Cosmos3-Nano-Policy-DROID"
    checkpoint_dir: str = "/local/model"          # weights staged locally before timing (§8)
    backend: str = "mock"                         # mock (no GPU) | vllm (target inference GPU)
    endpoint: str | None = None                   # vllm: externally-deployed endpoint (else launched)
    input_manifest: str = "policy/mock/manifest.json"
    output_dir: str = "results"
    torchinductor_root: str = "/tmp/torchinductor"   # per-config subdir /tmp/torchinductor/<cid>
    configurations: tuple = ()                    # cids to run; empty = full matrix (P0-P3,E0-E6)
    baseline_at_start_and_end: bool = True
    randomize_order: bool = True
    order_seed: int = 7
    wait_between_seconds: float = 5.0
    baseline_drift_reject_pct: float = 10.0       # reject if the two E0 baselines differ by more


@dataclass(frozen=True)
class DatasetConfig:
    """DROID observation shapes (static → CUDA-graph friendly, §9) + RoboLab quality subset
    structure (§5). Shapes are fixed across the replay set; the subset is 3×3×2 stratified."""
    camera_views: tuple = ("exterior", "wrist")   # DROID convention: exterior + wrist
    image_hw: tuple = (180, 320)                  # per-view RGB fed to the reasoner (VERIFY on-box)
    proprio_dim: int = 8                          # joint pos/vel + gripper
    instruction_tokens: int = 32                  # tokenized language-instruction length (bucketed)
    action_chunk: tuple = (32, 8)                 # the only evaluated task (§1): 32 steps × 8 DoF
    replay_size: int = 50                         # MEASURED requests/config = the 50 unique obs, once each
    replay_seed: int = 20260713                   # deterministic → reproducible logs (§10)
    capability_groups: tuple = ("visual", "relational", "procedural")   # RoboLab attribute families
    difficulty_levels: tuple = ("easy", "medium", "hard")
    tasks_per_cell: int = 2                        # 3 groups × 3 difficulty × 2 = 18 tasks
    episodes_per_task: int = 10


@dataclass(frozen=True)
class GeneratorSampling:
    """Generator (action-diffusion) sampling recipe for Cosmos3-Nano-Policy-DROID.

    Model-level inference hyperparameters, NOT per-rung optimization knobs: the whole
    generator/end-to-end waterfall samples with the SAME recipe, so the optimization
    techniques — and not a changed schedule — explain the latency deltas. `steps` is the
    denoising-loop length and therefore a static shape (§9) the CUDA-graph rungs capture;
    `denoising_step_ms` is an array of this length. Logged into environment.json (§10)."""
    steps: int = 4                      # denoising / flow-matching steps per action chunk
    guidance: float = 3.0              # classifier-free guidance scale
    shift: float = 5.0                 # flow-matching timestep-schedule shift
    cfg_mode: str = "full-range-null"  # CFG over the full step range against a null (uncond) branch

    @property
    def uses_cfg(self) -> bool:
        """CFG active (a real CFG mode with guidance>1) -> the CFG-Parallel multi-GPU
        experiment applies (§3, policy/multigpu.py)."""
        return self.cfg_mode != "none" and self.guidance > 1.0


@dataclass(frozen=True)
class ReasonerSampling:
    """Reasoner (Qwen3-VL conditioning) decode parameters for Cosmos3-Nano-Policy-DROID.

    The Reasoner is measured ONLY as action-policy conditioning — it does NOT generate
    standalone text (specification_revised.txt §2). Decoding is deterministic (temperature 0)
    so the conditioning — and therefore the logged action chunk — is reproducible (§10). These
    are the vLLM SamplingParams for the conditioning pass; VERIFY the canonical values on-box,
    especially `max_tokens` (the conditioning-token budget per observation)."""
    max_tokens: int = 256          # conditioning-token budget per observation (VERIFY on-box)
    temperature: float = 0.0       # greedy / deterministic conditioning (reproducible, §10)
    top_p: float = 1.0             # no nucleus truncation (inactive at temperature 0)
    top_k: int = -1                # disabled (vLLM convention)
    repetition_penalty: float = 1.0  # none (1.0 == off)

    @property
    def greedy(self) -> bool:
        """Deterministic decode (temperature 0) -> reproducible conditioning (§10)."""
        return self.temperature == 0.0


@dataclass(frozen=True)
class MeasurementConfig:
    """§6 + latency best practice: batch 1, ~50 warm-ups (excluded), measured = replay_size (the
    50 unique real obs, once each), p50/p90/p99 summaries (p99 rough at n=50)."""
    warmup_requests: int = 50
    min_measured_requests: int = 50
    percentiles: tuple = (50, 90, 99)


@dataclass(frozen=True)
class QualityGateConfig:
    """Lossy-technique (Cache-DiT / FP8) accept/reject thresholds (§9)."""
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
    """Load the single config file into `Settings` (code defaults fill any absent section/key,
    so the harness runs with no YAML at all). Rejects unknown sections/keys to catch typos."""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return Settings()
    data = yaml.safe_load(p.read_text()) or {}
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}; known: {sorted(_SECTIONS)}")
    return Settings(**{name: _build(cls, data.get(name) or {}) for name, cls in _SECTIONS.items()})


# The loaded singleton every module reads from. Import-safe (only stdlib + yaml).
CONFIG: Settings = load_config()
