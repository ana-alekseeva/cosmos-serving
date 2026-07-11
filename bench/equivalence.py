"""Losslessness checks for Part-2 techniques (specification.md §5, F8).

Proves a technique changed *speed* but not *output*: exact-match for autoregressive
tokens, tight numerical tolerance on latents, and SSIM~=1.0 on decoded frames.
Stubs here define the contract; real tensors are compared on the GPU in Part 2.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EquivalenceReport:
    kind: str
    metric: float
    threshold: float
    lossless: bool


def assert_token_equivalence(ref_tokens, new_tokens) -> EquivalenceReport:
    """Autoregressive path: token id sequences must match exactly."""
    mismatches = sum(1 for a, b in zip(ref_tokens, new_tokens) if a != b)
    mismatches += abs(len(ref_tokens) - len(new_tokens))
    return EquivalenceReport("token-exact", float(mismatches), 0.0, mismatches == 0)


def latent_max_abs_diff(ref, new, threshold: float = 1e-3) -> EquivalenceReport:
    """Diffusion path: max |ref - new| over the latent tensor (needs numpy/torch)."""
    import numpy as np
    diff = float(np.max(np.abs(np.asarray(ref) - np.asarray(new))))
    return EquivalenceReport("latent-max-abs-diff", diff, threshold, diff <= threshold)


def frame_ssim(ref_frames, new_frames, threshold: float = 0.99) -> EquivalenceReport:
    """Decoded video: mean SSIM across frames (~1.0 == lossless). Stub."""
    raise NotImplementedError(
        "Wire SSIM on the GPU in Part 2 (e.g. skimage.metrics.structural_similarity "
        "per frame, averaged); compare against threshold."
    )
