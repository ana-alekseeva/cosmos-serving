"""Shared data contract for the latency harness.

Backend-agnostic: this module defines the request/task TYPES and the manifest LOADER that
BOTH the mock and the real (vLLM/vLLM-Omni) paths consume. It holds no synthetic data — the
committed `manifest.json` is the source of truth for the replay set (instructions, ids,
shapes live there, not in code).

  1. Offline replay set — a FIXED set of requests captured from RoboLab episodes, loaded
     from `manifest.json`. Each request carries camera observations, instruction,
     proprioceptive state, task/episode ids, control timestep, and a fixed inference seed.
     The real driver rebuilds tensors from `capture_ref`; the mock reads only the shape
     metadata (latency is shape-driven). Each observation is measured once by default;
     `tile_to` can cycle the set to a larger measured count for tighter tail percentiles.

  2. RoboLab quality subset — a stratified ~18-task subset (3 capability groups x 3
     difficulty levels x 2 tasks), 10 episodes/task. Used to REJECT optimizations that
     damage policy performance (the lossy quality gate). The full RoboLab benchmark is
     only needed for baseline + final (Job 4).

The MOCK generator that PRODUCES a synthetic manifest for tests/dev lives in
`policy/mock/replay.py`; it is not imported here and never runs on the real path.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from policy.config import CONFIG

# All shapes/structure come from the single config file (config/experiment.yaml -> CONFIG).
# These module names are kept as thin re-exports so callers read the same symbols. DROID
# shapes are static across the replay set so CUDA-graph configs capture fixed shapes.
_D = CONFIG.dataset
CAMERA_VIEWS = _D.camera_views          # DROID convention (exterior + wrist)
IMAGE_HW = _D.image_hw                  # per-view RGB resolution fed to the reasoner
PROPRIO_DIM = _D.proprio_dim            # proprioceptive state dim (joint pos/vel + gripper)
INSTRUCTION_TOKENS = _D.instruction_tokens   # tokenized language-instruction length (bucketed)

DEFAULT_REPLAY_SIZE = _D.replay_size    # measured requests/config (= the unique replay set)

# RoboLab quality subset structure: 3 capability groups x 3 difficulty x 2 tasks = 18.
CAPABILITY_GROUPS = _D.capability_groups
DIFFICULTY_LEVELS = _D.difficulty_levels
TASKS_PER_CELL = _D.tasks_per_cell
EPISODES_PER_TASK = _D.episodes_per_task


@dataclass(frozen=True)
class DroidRequest:
    """One captured control step — the unit of the offline replay set."""
    request_id: int
    task: str
    episode_id: int
    control_timestep: int          # step index within the episode
    seed: int                      # fixed inference seed (reproducibility)
    instruction: str
    # Fixed shapes (static — CUDA-graph friendly). The mock uses only these; the real
    # driver materializes the actual tensors from the RoboLab capture referenced here.
    camera_views: tuple = CAMERA_VIEWS
    image_hw: tuple = IMAGE_HW
    proprio_dim: int = PROPRIO_DIM
    instruction_tokens: int = INSTRUCTION_TOKENS
    capture_ref: str = ""          # path/URI to the raw captured tensors (real backend)

    def as_dict(self) -> dict:
        return asdict(self)


def write_manifest(reqs: list[DroidRequest], path: str | Path, *,
                   source: str = "cosmos-droid-replay") -> Path:
    """Serialize a replay set to a manifest.json (the schema load_manifest reads).

    Shared by the mock generator (policy/mock/replay.py) and the real DROID capture
    (policy/capture.py). `static_shapes` is recorded from the first request's actual shapes
    so a real-capture manifest reports its true geometry."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    r0 = reqs[0] if reqs else None
    static = {} if r0 is None else {
        "camera_views": list(r0.camera_views), "image_hw": list(r0.image_hw),
        "proprio_dim": r0.proprio_dim, "instruction_tokens": r0.instruction_tokens,
        "action_chunk": list(CONFIG.dataset.action_chunk),
    }
    p.write_text(json.dumps({
        "dataset": source,
        "task": "DROID obs + instruction + proprio -> 32x8 action chunk",
        "count": len(reqs),
        "static_shapes": static,
        "requests": [r.as_dict() for r in reqs],
    }, indent=2))
    return p


def load_manifest(path: str | Path) -> list[DroidRequest]:
    """Load the fixed replay set from a manifest (shared by the mock and real paths).

    The committed manifest is the source of truth. Regenerate a synthetic one with
    `python -m policy.mock.replay`; a real run stages a manifest of real captures."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"replay manifest not found: {p}. Commit one, stage a real capture manifest, "
            f"or regenerate the mock fixture with `python -m policy.mock.replay`.")
    data = json.loads(p.read_text())
    out = []
    for r in data["requests"]:
        out.append(DroidRequest(
            request_id=r["request_id"], task=r["task"], episode_id=r["episode_id"],
            control_timestep=r["control_timestep"], seed=r["seed"],
            instruction=r["instruction"],
            camera_views=tuple(r.get("camera_views", CAMERA_VIEWS)),
            image_hw=tuple(r.get("image_hw", IMAGE_HW)),
            proprio_dim=r.get("proprio_dim", PROPRIO_DIM),
            instruction_tokens=r.get("instruction_tokens", INSTRUCTION_TOKENS),
            capture_ref=r.get("capture_ref", ""),
        ))
    return out


def tile_to(requests: list[DroidRequest], n: int) -> list[DroidRequest]:
    """Return exactly `n` measured requests from the fixed replay set.

    Default: `n == len(requests)`, so this is a pass-through — every unique observation is
    measured once. `n < len` takes the first `n` (e.g. a smoke run's replay_size=1). `n > len`
    cycles the set (raise replay_size if you want repeats for tighter tail percentiles).
    Deterministic (reproducible): `request_id` is the measured-slot index and the seed is
    re-derived per repeat, so any cycled requests are distinct but a pure function of input."""
    if not requests:
        return []
    out = []
    m = len(requests)
    for i in range(n):
        base = requests[i % m]
        rep = i // m
        seed = base.seed if rep == 0 else (base.seed * 2654435761 + rep) & 0x7FFFFFFF
        out.append(replace(base, request_id=i, seed=seed))
    return out


# ---------------------------------------------------------------------------
# RoboLab quality subset — stratified 3x3x2 = 18 tasks, 10 episodes each.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityTask:
    task: str
    capability: str
    difficulty: str
    episodes: int = EPISODES_PER_TASK


def quality_subset() -> list[QualityTask]:
    """The stratified 18-task subset used to reject optimizations that hurt policy success."""
    out = []
    for cap in CAPABILITY_GROUPS:
        for diff in DIFFICULTY_LEVELS:
            for t in range(TASKS_PER_CELL):
                out.append(QualityTask(f"RoboLab-{cap}-{diff}-{t}", cap, diff))
    return out
