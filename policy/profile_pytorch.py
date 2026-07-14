"""torch.profiler latency attribution for the native PyTorch backend.

Same technique as the gpu_and_inference_hw/hw2 profiler: run one real Cosmos-policy request
under `torch.profiler` and emit two things —

  1. a summary table sorted by CUDA time — WHERE the GPU time actually goes (attention, MLP,
     tokenizer, VAE, ...), attributed automatically per operator/module. No forward-hook
     submodule names required (this is why it works where the hooks returned reasoner_ms=0).
  2. a Chrome trace (open at https://ui.perfetto.dev) — the DASHBOARD: the reasoner phase runs
     once at the start, then the 4 denoising steps repeat, so the reasoner/denoise split is
     visible in the timeline, and `with_modules=True` labels each op with its module (which
     also tells you the exact reasoner/denoiser submodule names to wire into pytorch_engine).

Runs in cosmos-framework's env (like `--backend pytorch`), against a captured replay set +
the local checkpoint.

    python -m policy.profile_pytorch --configuration P1 \
      --manifest /local/replay/manifest.json --checkpoint-dir /local/model \
      --out results-smoke/trace_R1.json --warmups 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from policy.configs import config_by_id
from policy.dataset import load_manifest, tile_to
from policy.pipeline import make_engine


def profile_config(configuration: str, *, manifest: str, checkpoint_dir: str, out: str,
                   warmups: int = 3, rows: int = 25) -> Path:
    """Profile one request of `configuration` on the native PyTorch backend."""
    import torch

    cfg = config_by_id(configuration)
    engine = make_engine("pytorch", cfg, checkpoint_dir=checkpoint_dir)
    engine.prepare()
    req = tile_to(load_manifest(manifest), 1)[0]

    for _ in range(warmups):                 # warm compile / CUDA graphs / caches OUT of the trace
        engine.run_request(req)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_modules=True,                   # attribute each op to its nn.Module (reasoner vs DiT)
    ) as prof:
        engine.run_request(req)
    torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=rows))
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out_path))
    print(f"\nChrome trace -> {out_path}   (open at https://ui.perfetto.dev)")
    print("In the timeline: the reasoner runs once, then the 4 denoising steps repeat — that "
          "temporal split is your reasoner_ms vs denoising_ms.")
    engine.close()
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="torch.profiler latency attribution (native PyTorch backend).")
    ap.add_argument("--configuration", required=True, help="config id, e.g. P0 / P1 / E2")
    ap.add_argument("--manifest", default="/local/replay/manifest.json")
    ap.add_argument("--checkpoint-dir", default="/local/model")
    ap.add_argument("--out", default="results-smoke/trace.json", help="Chrome trace output path")
    ap.add_argument("--warmups", type=int, default=3, help="warm-up requests before the profiled one")
    ap.add_argument("--rows", type=int, default=25, help="rows in the summary table")
    a = ap.parse_args()
    profile_config(a.configuration, manifest=a.manifest, checkpoint_dir=a.checkpoint_dir,
                   out=a.out, warmups=a.warmups, rows=a.rows)


if __name__ == "__main__":
    main()
