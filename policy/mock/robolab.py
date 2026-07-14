"""Modeled RoboLab task success (MOCK — specification_revised.txt §4 Jobs 3-4, §5).

The real RoboLab gate drives Isaac Sim / Isaac Lab on an RT-core GPU and measures task
success from executing the generated actions (policy/robolab.py::_run_real_robolab). This
module models per-task success offline so the gate logic + report render with no simulator:
the lossy techniques (Cache-DiT, FP8) apply a small success penalty that stays below the
rejection threshold, and difficulty sets the baseline success rate.
"""
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
