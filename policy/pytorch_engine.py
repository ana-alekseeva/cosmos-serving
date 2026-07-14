"""Native PyTorch reference backend (Cosmos 3 report §5.3.1).

This is the *modifiable reference* serving path the report describes: it runs the model and
the surrounding inference procedure directly in eager-mode PyTorch, mirroring the training
computation, and is the primary target for landing/validating optimizations BEFORE porting
them to the production runtimes (vLLM / vLLM-Omni). It is the backend the latency waterfall
(§4 Job 1) measures; Job 2 re-validates the final config on vLLM-Omni.

Workflow (§5.3.1), timed per stage with CUDA events:
  1. Input preparation  — build the modality conditioning dicts + input tensors, host->device.
  2. Reasoner           — conditioning for the Generator. Invariant across the denoising
                          trajectory, so with reasoner caching it is computed ONCE/observation
                          (§5.3.1 Reasoner-tower caching); the naive path recomputes per step.
  3. Diffusion loop     — flow-matching timestep schedule, per-step denoiser call, classifier-
                          free guidance (two forward passes combined), sampler update.
  4. Decode/post-process — latent -> 32x8 action chunk, denormalize/clip, device->host.

Optimizations implemented here are ONLY the §5.3.1 set (see policy/compat.py):
  * attention backend    — forced math vs flash SDPA (§9: fail, do not silently fall back).
  * torch.compile        — regional compilation of the repeated transformer blocks
                           ("transformer-layer granularity", §5.3.1); the Python serving loop
                           stays eager.
  * CUDA-graph replay     — torch.compile(mode="reduce-overhead") on those blocks.
  * Reasoner-tower caching — compute conditioning once/observation, reuse across steps.

Cache-DiT and FP8 are vLLM-Omni features (§5.3.3) and are REJECTED here by compat.validate —
the native waterfall stops at the Reasoner cache; those rungs run on vLLM-Omni.

Only the model load + the model's forward signatures are `# VERIFY` hooks (the weights are not
in this repo); the orchestration, optimization application, and timing are complete.
"""
from __future__ import annotations

import hashlib
import time

from policy import compat
from policy.configs import GENERATOR_SAMPLING, N_DENOISE_STEPS, Config
from policy.dataset import DroidRequest
from policy.measure import LatencyRecord

_ACTION_CHUNK = (32, 8)          # the only evaluated task (§1)


def _load_policy(checkpoint_dir: str, *, device, dtype):
    """Load the Cosmos3-Nano-Policy-DROID PyTorch model in eval/bf16 on `device`.

    VERIFY on-box: the model package + entry point, and the forward interface this engine
    calls below:
        policy.reason(obs)            -> conditioning (once/observation, cacheable)
        policy.reason_null(n)         -> unconditional conditioning for CFG
        policy.denoiser(x, sigma, c)  -> velocity/noise prediction
        policy.denoiser.blocks        -> the repeated transformer blocks (compiled per §5.3.1)
        policy.init_latent(cond)      -> initial noised action latent
        policy.decode_action(x)       -> (32, 8) action chunk (denormalized)
    """
    raise NotImplementedError(
        f"load Cosmos3-Nano-Policy-DROID from {checkpoint_dir!r} in eager bf16. VERIFY the model "
        "entry point and the reason()/denoiser()/init_latent()/decode_action() interface used "
        "by PyTorchPolicyEngine (policy/pytorch_engine.py).")


class PyTorchPolicyEngine:
    """Native eager-PyTorch reference path (§5.3.1). Backend id: 'pytorch'."""

    backend = "pytorch"

    def __init__(self, config: Config, *, model: str | None = None,
                 checkpoint_dir: str | None = None, **_):
        compat.validate(config, "pytorch")          # refuse Cache-DiT/FP8 up front (§5.3.3)
        self.config = config
        self.model = model or "nvidia/Cosmos3-Nano-Policy-DROID"
        self.checkpoint_dir = checkpoint_dir or "/local/model"
        self._policy = None
        self._device = None

    # -- lifecycle ----------------------------------------------------------------
    def prepare(self) -> None:
        import torch

        self._device = torch.device("cuda")
        self._policy = _load_policy(self.checkpoint_dir, device=self._device,
                                    dtype=torch.bfloat16)
        self._apply_compile()                        # torch.compile / CUDA graphs (§5.3.1)

    def _sdpa_backend(self):
        """Forced SDPA backend (§9): flash if requested (fail, don't fall back), else math."""
        from torch.nn.attention import SDPBackend
        return (SDPBackend.FLASH_ATTENTION
                if self.config.stage_flags.get("attention") == "flash"
                else SDPBackend.MATH)

    def _apply_compile(self) -> None:
        """Regional compilation of the repeated transformer blocks (§5.3.1).

        reduce-overhead == compile + CUDA-graph replay; default == compile without graphs;
        neither == eager. The outer serving loop (schedule, CFG, sampler) stays in Python."""
        import torch

        flags = self.config.stage_flags
        if flags.get("cuda_graphs"):
            mode = "reduce-overhead"                 # compile + CUDA graphs (combined config, §9)
        elif flags.get("compile"):
            mode = "default"                         # compile, no graphs
        else:
            return                                   # eager baseline
        blocks = getattr(self._policy.denoiser, "blocks", None)   # VERIFY attribute name
        if blocks is None:                           # fall back to whole-denoiser compile
            self._policy.denoiser = torch.compile(self._policy.denoiser, mode=mode, fullgraph=True)
        else:
            for i in range(len(blocks)):             # compile each repeated block (regional)
                blocks[i] = torch.compile(blocks[i], mode=mode, fullgraph=True)

    def close(self) -> None:
        self._policy = None

    # -- one request (§5.3.1 workflow, CUDA-event timed) --------------------------
    def run_request(self, req: DroidRequest) -> LatencyRecord:
        import torch
        from torch.nn.attention import sdpa_kernel

        dev = self._device
        cached = self.config.reasoner_cached
        gen = GENERATOR_SAMPLING
        torch.cuda.reset_peak_memory_stats(dev)

        def _evpair():
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        pre, rea, prep, post, d2h = _evpair(), _evpair(), _evpair(), _evpair(), _evpair()
        step_evs = [_evpair() for _ in range(N_DENOISE_STEPS)]

        with torch.inference_mode(), sdpa_kernel(self._sdpa_backend()):
            # 1. Input preparation (host tensors -> device). VERIFY the conditioning builder.
            pre[0].record()
            obs = self._prepare_inputs(req, dev)
            pre[1].record()

            # 2. Reasoner conditioning — once if cached, else recomputed per denoising step.
            rea[0].record()
            cond = uncond = None
            if cached:
                cond, uncond = self._reason(obs)     # invariant across the trajectory (§5.3.1)
            rea[1].record()

            # 3. Generator preparation — schedule + initial latent.
            prep[0].record()
            sigmas = self._sigmas(gen.steps, gen.shift, dev)
            x = self._policy.init_latent(cond if cached else obs)   # VERIFY
            prep[1].record()

            # 4. Diffusion loop — per-step denoiser + CFG + sampler update.
            reasoner_extra = 0.0                     # per-step reasoner cost when NOT cached
            for i in range(gen.steps):
                if not cached:                       # naive path recomputes conditioning/step
                    r0, r1 = _evpair()
                    r0.record(); cond, uncond = self._reason(obs); r1.record()
                    step_evs[i] = (step_evs[i][0], step_evs[i][1], r0, r1)  # carry reasoner evs
                step_evs[i][0].record()
                x = self._denoise_step(x, sigmas[i], sigmas[i + 1], cond, uncond, gen.guidance)
                step_evs[i][1].record()

            # 5. Decode + post-process, then device -> host.
            post[0].record()
            action = self._policy.decode_action(x)   # (32, 8) VERIFY
            post[1].record()
            d2h[0].record()
            action_cpu = action.detach().to("cpu", non_blocking=False)
            d2h[1].record()

        torch.cuda.synchronize(dev)                  # single sync, then read all timers (§6)

        denoise_steps = [a.elapsed_time(b) for (a, b, *_) in step_evs]
        reasoner_ms = rea[0].elapsed_time(rea[1])
        if not cached:                               # add the per-step recomputation cost
            reasoner_ms += sum(r0.elapsed_time(r1) for (*_, r0, r1) in step_evs)
        preprocess_ms = pre[0].elapsed_time(pre[1])
        gen_prep_ms = prep[0].elapsed_time(prep[1])
        postprocess_ms = post[0].elapsed_time(post[1])
        d2h_ms = d2h[0].elapsed_time(d2h[1])
        denoising_ms = sum(denoise_steps)
        server_ms = (preprocess_ms + reasoner_ms + gen_prep_ms + denoising_ms
                     + postprocess_ms + d2h_ms)
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

        return LatencyRecord(
            request_id=req.request_id, task=req.task, episode_id=req.episode_id,
            preprocess_ms=preprocess_ms, h2d_ms=0.0, reasoner_ms=reasoner_ms,
            generator_prepare_ms=gen_prep_ms, denoising_ms=denoising_ms,
            denoising_step_ms=denoise_steps, postprocess_ms=postprocess_ms, d2h_ms=d2h_ms,
            server_ms=server_ms, transport_ms=0.0,       # in-process: no client/server transport
            first_action_ms=server_ms, total_chunk_ms=server_ms,
            peak_memory_mb=peak_mb,
            output_checksum=_action_checksum(action_cpu),
            quality_gate="n/a",                          # pytorch runs only lossless configs
        )

    # -- model hooks (VERIFY the real signatures on-box) --------------------------
    def _prepare_inputs(self, req: DroidRequest, dev):
        """Build the conditioning dict + input tensors from the captured observation.

        VERIFY: load the real DROID tensors (policy.capture.load_capture on req.capture_ref),
        normalize, and assemble the model's modality conditioning dict on `dev`."""
        from policy.capture import load_capture
        obs = load_capture(req.capture_ref)          # exterior/wrist/proprio/instruction
        return self._policy.build_conditioning(obs, device=dev, seed=req.seed)   # VERIFY

    def _reason(self, obs):
        """Reasoner conditioning for CFG: (conditional, unconditional). VERIFY signatures."""
        return self._policy.reason(obs), self._policy.reason_null(obs)

    def _denoise_step(self, x, sigma, sigma_next, cond, uncond, guidance):
        """One flow-matching Euler step with classifier-free guidance (full-range null, §1).

        CFG needs two forwards — conditional + unconditional — combined per step (§5.3.1
        Diffusion loop). VERIFY the denoiser signature and the sampler update convention."""
        v_cond = self._policy.denoiser(x, sigma, cond)
        v_uncond = self._policy.denoiser(x, sigma, uncond)
        v = v_uncond + guidance * (v_cond - v_uncond)        # full-range CFG null
        return x + (sigma_next - sigma) * v                  # Euler update; VERIFY convention

    @staticmethod
    def _sigmas(steps: int, shift: float, dev):
        """Flow-matching timestep schedule with shift (Cosmos/SD3-style). VERIFY the exact recipe."""
        import torch
        t = torch.linspace(1.0, 0.0, steps + 1, device=dev)
        return shift * t / (1.0 + (shift - 1.0) * t)


def _action_checksum(action) -> str:
    """Checksum of the (32,8) action chunk, rounded to tolerate compile/kernel FP noise so
    lossless rungs (flash/compile/graphs/reasoner-cache) match the eager baseline (§10)."""
    import numpy as np
    a = np.round(np.asarray(action, dtype="float64"), 3)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]
