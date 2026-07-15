"""Synthetic replay-set generator (MOCK).

Produces the committed `policy/mock/manifest.json` fixture so the harness has a fixed,
reproducible replay set to run against with no GPU and no real captures. The RUNTIME never
imports this: run_matrix / run_configuration / multigpu load the committed manifest via
`policy.dataset.load_manifest`. This module only (re)generates that file.

    python -m policy.mock.replay                 # regenerate policy/mock/manifest.json
    python -m policy.mock.replay out.json 128    # custom path + unique-request count

A real run replaces the fixture with a manifest of real RoboLab captures (real pixels /
proprio behind each `capture_ref`); the shapes and schema are identical.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from policy.config import CONFIG
from policy.dataset import (
    CAPABILITY_GROUPS,
    DEFAULT_REPLAY_SIZE,
    DIFFICULTY_LEVELS,
    TASKS_PER_CELL,
    DroidRequest,
    write_manifest,
)

DEFAULT_REPLAY_SEED = CONFIG.dataset.replay_seed   # deterministic -> reproducible logs
FIXTURE_SIZE = 50                                  # unique observations in the committed fixture
# The committed synthetic fixture lives next to this generator (policy/mock/manifest.json),
# so it reads as mock data — not a real dataset. The on-box run stages a manifest to the
# path in config run.input_manifest (/local/... on the Nebius GPU box); a real run replaces
# the fixture there with real RoboLab captures.
_FIXTURE_PATH = Path(__file__).resolve().parent / "manifest.json"

# Instruction pool for the synthetic fixture. Real captures carry the episode's own
# instruction; these stand in so the mock set has representative language variety.
_INSTRUCTIONS = (
    "pick up the red cube and place it in the bowl",
    "open the top drawer",
    "wipe the table with the sponge",
    "stack the blue block on the green block",
    "close the microwave door",
    "pour the beads into the cup",
    "insert the peg into the hole",
    "hang the towel on the rack",
    "put the banana on the plate",
    "turn the knob clockwise",
    "place the mug upright on the shelf",
    "sort the red and blue blocks into bins",
)


def build_mock_replay(n: int = DEFAULT_REPLAY_SIZE, *,
                      seed: int = DEFAULT_REPLAY_SEED) -> list[DroidRequest]:
    """Deterministic synthetic replay set of `n` unique requests across tasks/episodes.

    Shapes are constant (static-shape requirement); task/episode/instruction vary so
    the set is representative of real preprocessing, prompt lengths, and memory layouts.
    Deterministic in `seed` so the committed fixture is reproducible."""
    rng = random.Random(seed)
    reqs: list[DroidRequest] = []
    tasks = [f"RoboLab-{g}-{d}-{t}" for g in CAPABILITY_GROUPS
             for d in DIFFICULTY_LEVELS for t in range(TASKS_PER_CELL)]
    rid = 0
    while rid < n:
        task = tasks[rid % len(tasks)]
        episode_id = rid % 8
        for step in range(4):                      # ~4 control steps sampled per (task, episode)
            if rid >= n:
                break
            reqs.append(DroidRequest(
                request_id=rid, task=task, episode_id=episode_id,
                control_timestep=step, seed=rng.randint(1, 2**31 - 1),
                instruction=_INSTRUCTIONS[rid % len(_INSTRUCTIONS)],
                capture_ref=f"robolab://{task}/ep{episode_id}/step{step}",
            ))
            rid += 1
    return reqs


def write_mock_manifest(path: str | Path = _FIXTURE_PATH, *, n: int = FIXTURE_SIZE,
                        seed: int = DEFAULT_REPLAY_SEED) -> Path:
    """Build a synthetic replay set and write it to `path` (the committed fixture default)."""
    return write_manifest(build_mock_replay(n, seed=seed), path, source="cosmos-droid-mock")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _FIXTURE_PATH
    count = int(sys.argv[2]) if len(sys.argv) > 2 else FIXTURE_SIZE
    written = write_mock_manifest(out, n=count)
    print(f"wrote {count} requests -> {written}")
