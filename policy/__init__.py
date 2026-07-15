"""Cosmos3-Nano-Policy-DROID serving harness.

Latency attribution + optimization for the *only* evaluated inference task:

    DROID camera observations + language instruction + proprioceptive state
        -> 32 x 8 robot-action chunk

The model is evaluated by executing its generated actions in RoboLab. This package
implements the PyTorch ablation matrix (native P0-P3 (merged reasoner+generator), combined
end-to-end), the fixed offline replay set, the per-request latency logs, the waterfall
/ stage-breakdown figures, and the aggregation job.

The harness runs end-to-end on the `mock` backend (a modeled per-stage latency table
anchored to the spec's example log) so the plumbing, logs, and figures are validated
before touching a GPU. Swap `--backend mock` -> `--backend vllm` on the target GPU.
"""
