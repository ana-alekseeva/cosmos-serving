"""Modeled RoboLab task success (MOCK): offline per-task success so the gate renders with no simulator."""
from __future__ import annotations

import random
import zlib

from policy.configs import Config
from policy.dataset import QualityTask

# Modeled baseline success by difficulty (mock only). The real number comes from RoboLab.
_BASE_SUCCESS = {"easy": 0.88, "medium": 0.72, "hard": 0.54}
# Modeled success penalty from a lossy technique's action drift (mock). Below the threshold.
_LOSSY_PENALTY = {"cache_dit": 0.006, "quantization": 0.012}


def config_success(task: QualityTask, config: Config, *, seed: int) -> float:
    """Modeled success rate for one (task, config): difficulty baseline minus lossy penalty,
    with deterministic per-task episode noise."""
    base = _BASE_SUCCESS[task.difficulty]
    penalty = sum(p for k, p in _LOSSY_PENALTY.items() if config.stage_flags.get(k))
    rng = random.Random(zlib.crc32(f"{task.task}|{config.cid}|{seed}".encode()))
    noise = (rng.random() - 0.5) * 0.04                # +/-2 points episode-to-episode noise
    return max(0.0, min(1.0, base - penalty + noise))
