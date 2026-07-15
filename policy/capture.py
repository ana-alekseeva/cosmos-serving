"""Capture a REAL DROID replay set for latency measurement.

Pulls N real observations from the open DROID dataset (Khazatsky et al. 2024) and writes them
to disk + a manifest that `policy.dataset.load_manifest` reads — so the REAL run (Jobs 1 & 2)
replays genuine robot observations instead of the synthetic `policy/mock/manifest.json`. Both
jobs stage this ONE manifest, so their inputs are identical (comparability).

DROID is exactly the observation format Cosmos3-Nano-Policy-DROID consumes. We capture the
three RGB views the policy's RoBoArena concat view is built from (`wrist_image_left` +
`exterior_image_1_left` + `exterior_image_2_left`) plus `joint_position` (7) +
`gripper_position` (1) as the 8-D proprio state and the language instruction.

Why 50 observations: at batch-1 with fixed shapes, latency is ~content-independent, so a small
fixed set is representative. We measure each of the 50 once (replay_size=50) — solid p50, ok
p90, rough p99. Raise replay_size to cycle the set (`dataset.tile_to`) for tighter tails,
MLPerf single-stream style (a fixed set repeated to the query count).

Runs on a box with `tensorflow-datasets` + `tensorflow` and DROID access. droid_100 is a small
~real subset (100 episodes / 32k frames / 47 tasks) loaded by DIRECTORY from the public bucket
gs://gresearch/robotics/droid_100/<ver> (NOT a registered tfds name). NOT exercised by the mock
tests; every DROID field name below is a `# VERIFY` against your build.

    uv pip install tensorflow-datasets tensorflow-cpu
    python -m policy.capture --n 50 --out data/replay_real
    # then run the real matrix against it:
    uv run python run_matrix.py --backend vllm --input-manifest data/replay_real/manifest.json
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from policy.config import CONFIG
from policy.dataset import DroidRequest, write_manifest

FIXTURE_SIZE = 50
STEPS_PER_EPISODE = 3            # spread the N observations across ~N/3 episodes for variety
# droid_100 is NOT a registered tfds name — it is an Open X-Embodiment RLDS dataset loaded by
# DIRECTORY from the public GCS bucket (a ~2GB, 100-trajectory sample in the full-droid format).
DROID_BUILDER_DIR = "gs://gresearch/robotics/droid_100/1.0.0"   # VERIFY the version subdir

# DROID RLDS field names (# VERIFY against your tfds `droid`/`droid_100` build).
_EXT_KEY = "exterior_image_1_left"   # exterior view 1 (RoBoArena 3-view concat)
_EXT2_KEY = "exterior_image_2_left"  # exterior view 2 (RoBoArena 3-view concat)
_WRIST_KEY = "wrist_image_left"      # wrist view       (RoBoArena 3-view concat)
_JOINT_KEY = "joint_position"        # 7 joint angles
_GRIPPER_KEY = "gripper_position"    # 1 gripper position
_INSTR_KEY = "language_instruction"  # per-step language instruction


def capture_droid(n: int = FIXTURE_SIZE, out_dir: str | Path = "data/replay_real", *,
                  dataset: str = DROID_BUILDER_DIR, seed: int = CONFIG.dataset.replay_seed) -> Path:
    """Pull `n` real DROID observations -> per-obs .npz tensors + a manifest at out_dir.

    Deterministic (fixed episode/step stride + seed) so the captured set is reproducible and
    IDENTICAL across Jobs 1 & 2. Static-shape requirement: observations whose
    image resolution differs from the first kept one are skipped (CUDA-graph rungs need a
    single bucketed shape) — logged so the drop is not silent."""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide TF INFO/WARNING (oneDNN, GCS auth fallback)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # drop the oneDNN notice
    import numpy as np
    try:
        import tensorflow_datasets as tfds
    except ImportError as e:  # pragma: no cover - box-only dependency
        raise SystemExit(
            "capture needs tensorflow-datasets + tensorflow (to read the RLDS tfrecords over gs://):\n"
            "  uv pip install tensorflow-datasets tensorflow-cpu") from e

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    # OXE datasets (droid / droid_100) load by DIRECTORY, not a registered tfds name.
    if "://" in dataset or "/" in dataset:
        ds = tfds.builder_from_directory(builder_dir=dataset).as_dataset(split="train")
    else:
        ds = tfds.load(dataset, split="train")      # a registered tfds name, if you have one

    reqs: list[DroidRequest] = []
    ref_hw: tuple | None = None
    skipped = 0
    for ep_idx, episode in enumerate(ds):
        step_idx = 0
        for step in episode["steps"]:               # VERIFY: nested per-step dataset
            if step_idx >= STEPS_PER_EPISODE:
                break
            obs = step["observation"]
            ext = obs[_EXT_KEY].numpy()             # (H, W, 3) uint8   # VERIFY key + dtype
            ext2 = obs[_EXT2_KEY].numpy()           # second exterior view (3-view concat)
            wrist = obs[_WRIST_KEY].numpy()
            joint = np.reshape(obs[_JOINT_KEY].numpy(), -1)      # (7,)
            gripper = np.reshape(obs[_GRIPPER_KEY].numpy(), -1)  # (1,)
            proprio = np.concatenate([joint, gripper]).astype("float32")   # 8-D
            instr = step[_INSTR_KEY].numpy()
            instr = instr.decode("utf-8") if isinstance(instr, bytes) else str(instr)

            hw = tuple(int(x) for x in ext.shape[:2])
            if ref_hw is None:
                ref_hw = hw
            if hw != ref_hw:                        # static shapes only
                skipped += 1
                step_idx += 1
                continue

            idx = len(reqs)
            npz = out / f"{idx:04d}.npz"
            np.savez_compressed(npz, exterior=ext, exterior_2=ext2, wrist=wrist,
                                proprio=proprio, instruction=instr)
            reqs.append(DroidRequest(
                request_id=idx, task=f"droid-ep{ep_idx}", episode_id=ep_idx,
                control_timestep=step_idx, seed=rng.randint(1, 2**31 - 1),
                instruction=instr, image_hw=hw, proprio_dim=int(proprio.shape[0]),
                capture_ref=str(npz.resolve()),
            ))
            step_idx += 1
            if len(reqs) >= n:
                break
        if len(reqs) >= n:
            break

    if len(reqs) < n:
        print(f"WARNING: captured {len(reqs)}/{n} observations "
              f"({skipped} skipped for shape mismatch) — dataset exhausted.")
    manifest = write_manifest(reqs, out / "manifest.json", source="droid-real-replay")
    print(f"captured {len(reqs)} real DROID observations ({skipped} skipped) -> {manifest}")
    return manifest


def load_capture(capture_ref: str) -> dict:
    """Load one real observation's tensors from a capture .npz (real serving path)."""
    import numpy as np
    d = np.load(capture_ref, allow_pickle=True)
    obs = {"exterior": d["exterior"], "wrist": d["wrist"],
           "proprio": d["proprio"], "instruction": str(d["instruction"])}
    if "exterior_2" in d.files:                     # 3-view concat (RoBoArena) captures
        obs["exterior_2"] = d["exterior_2"]
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture a real DROID replay set for latency runs.")
    ap.add_argument("--n", type=int, default=FIXTURE_SIZE, help="unique observations to capture")
    ap.add_argument("--out", type=Path, default=Path("data/replay_real"), help="output dir")
    ap.add_argument("--dataset", default=DROID_BUILDER_DIR,
                    help="RLDS builder dir (gs://gresearch/robotics/droid[_100]/<ver>) or a tfds name")
    ap.add_argument("--seed", type=int, default=CONFIG.dataset.replay_seed)
    args = ap.parse_args()
    capture_droid(args.n, args.out, dataset=args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()
