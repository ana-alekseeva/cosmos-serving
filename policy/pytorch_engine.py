"""Native PyTorch reference backend (Cosmos 3 report) via cosmos_framework.

Reuses RobolabPolicyService's `generate_samples_from_batch` per replay request, timed with CUDA
events. Forward hooks on the reasoner/denoiser submodules split the one call into
reasoner_ms/denoising_ms (opaque under CUDA-graph replay). FP8/Cache-DiT route to vllm, not here.
Requires cosmos_framework; every cosmos symbol is a `# VERIFY` against your version.
"""
from __future__ import annotations

import hashlib
import warnings

from policy import compat
from policy.configs import ACTION_CHUNK, GENERATOR_SAMPLING, Config
from policy.dataset import DroidRequest
from policy.measure import LatencyRecord

_ACTION_T, _ACTION_D = ACTION_CHUNK        # 32 x 8

# Submodule attribute names to time (VERIFY: print [n for n,_ in model.named_children()] on-box).
_REASONER_CANDIDATES = ("reasoner", "reasoner_tower", "text_encoder", "vlm", "llm")
_DENOISER_CANDIDATES = ("denoiser", "dit", "net", "generator", "diffusion_model")


class _StageHooks:
    """Time named submodules via forward hooks + CUDA events. Eager/compile only — graph replay is opaque."""

    def __init__(self):
        self._handles: list = []
        self._pending: dict = {}
        self.calls: dict = {}

    def attach(self, name: str, module) -> None:
        import torch

        def pre(_m, _inp):
            s = torch.cuda.Event(enable_timing=True); s.record(); self._pending[name] = s

        def post(_m, _inp, _out):
            e = torch.cuda.Event(enable_timing=True); e.record()
            self.calls.setdefault(name, []).append((self._pending.pop(name, None), e))

        self._handles += [module.register_forward_pre_hook(pre),
                          module.register_forward_hook(post)]

    def reset(self) -> None:
        self._pending, self.calls = {}, {}

    def elapsed(self, name: str) -> list[float]:            # call after cuda.synchronize()
        return [s.elapsed_time(e) for s, e in self.calls.get(name, []) if s is not None]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


def _find(model, names):
    for n in names:
        m = getattr(model, n, None)
        if m is not None:
            return m
    return None


def _build_service(checkpoint: str, config: Config, gen):
    """RobolabPolicyService subclassed to inject the compile / CUDA-graph flags."""
    from cosmos_framework.inference.args import OmniSetupOverrides                       # VERIFY
    from cosmos_framework.scripts.action_policy_server_robolab import (                  # VERIFY
        _DEFAULT_ROBOLAB_OUTPUT_DIR,
        RobolabPolicyService,
        RobolabServerArgs,
    )
    from cosmos_framework.scripts.action_policy_server_utils import (                     # VERIFY
        disable_runtime_ema_for_frozen_config,
    )

    flags = config.stage_flags
    compile_ = bool(flags.get("compile") or flags.get("cuda_graphs"))   # graphs imply compile
    graphs = bool(flags.get("cuda_graphs"))

    class _AblationService(RobolabPolicyService):
        def _build_setup_args(self, args):
            overrides = {
                "checkpoint_path": args.checkpoint_path,
                "output_dir": args.output_dir or _DEFAULT_ROBOLAB_OUTPUT_DIR,
                "sampler": args.sampler,
                # Skip the gated Guardrail download — content filter, not action-policy latency.
                "guardrails": False,
                "use_torch_compile": compile_,      # torch.compile (cosmos default is True!)
                "use_cuda_graphs": graphs,          # CUDA-graph replay
            }
            setup = OmniSetupOverrides.model_validate(overrides).build_setup()
            return disable_runtime_ema_for_frozen_config(setup)

    args = RobolabServerArgs(
        checkpoint_path=checkpoint, action_space="joint_pos",
        guidance=gen.guidance, num_steps=gen.steps, shift=gen.shift,
        deterministic_seed=True,        # per-request seed comes from the replay set instead
    )
    return _AblationService(args)


class PyTorchPolicyEngine:
    """Native eager-PyTorch reference path via cosmos_framework. Backend id: 'pytorch'."""

    backend = "pytorch"

    def __init__(self, config: Config, *, model: str | None = None,
                 checkpoint_dir: str | None = None, **_):
        compat.validate(config, "pytorch")           # refuse Cache-DiT/FP8
        self.config = config
        self.checkpoint = checkpoint_dir or model or "nvidia/Cosmos3-Nano-Policy-DROID"
        self._svc = None
        self._model = None
        self._device = None
        self._hooks = None

    def prepare(self) -> None:
        import os

        import torch

        # cuDNN *fused* attention (flash-class); needs cuDNN >= 9.22 but no flash_attn build.
        # I4_ATTN_BACKENDS is read at attention-call time; setdefault lets a box override it.
        os.environ.setdefault("I4_ATTN_BACKENDS", "cudnn")

        self._svc = _build_service(self.checkpoint, self.config, GENERATOR_SAMPLING)
        self._model = self._svc.model
        self._device = torch.device("cuda")
        # SKIP hooks under CUDA-graph replay: the per-forward CUDA-event records force cudagraph
        # partitioning, fragmenting the loop so replay can't remove launch overhead.
        self._hooks = _StageHooks()
        if self.config.stage_flags.get("cuda_graphs"):
            return
        reasoner = _find(self._model, _REASONER_CANDIDATES)
        denoiser = _find(self._model, _DENOISER_CANDIDATES)
        if reasoner is not None:
            self._hooks.attach("reasoner", reasoner)
        if denoiser is not None:
            self._hooks.attach("denoiser", denoiser)
        if reasoner is None or denoiser is None:
            warnings.warn(
                "stage hooks: set reasoner/denoiser submodule names in policy/pytorch_engine.py "
                f"— model children: {[n for n, _ in self._model.named_children()]}")

    def run_request(self, req: DroidRequest) -> LatencyRecord:
        import torch
        from cosmos_framework.scripts.action_policy_server_robolab import (   # VERIFY
            _build_data_batch_from_sample,
        )

        gen = GENERATOR_SAMPLING
        dev = self._device
        torch.cuda.reset_peak_memory_stats(dev)
        self._hooks.reset()

        def _ev():
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        pre, comp, post = _ev(), _ev(), _ev()
        with torch.inference_mode():
            pre[0].record()
            sample = self._svc._build_sample(self._obs(req))    # cosmos transform pipeline
            data_batch = _build_data_batch_from_sample(sample)
            pre[1].record()

            comp[0].record()
            samples = self._model.generate_samples_from_batch(  # reasoner + diffusion + decode
                data_batch, guidance=gen.guidance, seed=[req.seed],
                num_steps=gen.steps, shift=gen.shift)
            comp[1].record()

            post[0].record()
            cfg = self._svc.cfg
            action = samples["action"][0][:, :cfg.action_dim][cfg.history_length:]   # [32, 8]
            action_cpu = action.detach().to("cpu")
            post[1].record()

        torch.cuda.synchronize(dev)
        denoise_steps = self._hooks.elapsed("denoiser")
        reasoner_ms = sum(self._hooks.elapsed("reasoner"))
        denoising_ms = sum(denoise_steps)
        preprocess_ms = pre[0].elapsed_time(pre[1])
        compute_ms = comp[0].elapsed_time(comp[1])
        postprocess_ms = post[0].elapsed_time(post[1])
        gen_prep_ms = max(0.0, compute_ms - reasoner_ms - denoising_ms)   # remainder of the call
        server_ms = preprocess_ms + compute_ms + postprocess_ms
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

        return LatencyRecord(
            request_id=req.request_id, task=req.task, episode_id=req.episode_id,
            preprocess_ms=preprocess_ms, h2d_ms=0.0, reasoner_ms=reasoner_ms,
            generator_prepare_ms=gen_prep_ms, denoising_ms=denoising_ms,
            denoising_step_ms=denoise_steps, postprocess_ms=postprocess_ms, d2h_ms=0.0,
            server_ms=server_ms, transport_ms=0.0,       # in-process: no client/server transport
            first_action_ms=server_ms, total_chunk_ms=server_ms,
            peak_memory_mb=peak_mb,
            output_checksum=_action_checksum(action_cpu),
            quality_gate="n/a",                          # pytorch runs only lossless configs
        )

    def _obs(self, req: DroidRequest) -> dict:
        """Raw DROID observation dict for RobolabPolicyService._build_sample (3-view concat + proprio + prompt)."""
        import numpy as np
        from policy.capture import load_capture

        o = load_capture(req.capture_ref)
        proprio = np.asarray(o["proprio"], dtype=np.float32).reshape(-1)     # 8-D joint(7)+gripper(1)
        obs = {
            "prompt": o["instruction"],
            "observation/joint_position": proprio[:7][None, :],             # [1, 7]
            "observation/gripper_position": proprio[7:8][None, :],          # [1, 1]
        }
        if "exterior_2" in o:                                               # 3-view concat (trained view)
            obs["observation/wrist_image_left"] = np.asarray(o["wrist"], dtype=np.uint8)
            obs["observation/exterior_image_1_left"] = np.asarray(o["exterior"], dtype=np.uint8)
            obs["observation/exterior_image_2_left"] = np.asarray(o["exterior_2"], dtype=np.uint8)
        else:                                                               # older 2-view capture
            obs["observation/image"] = np.asarray(o["exterior"], dtype=np.uint8)
        return obs

    def close(self) -> None:
        if self._hooks is not None:
            self._hooks.remove()
        self._svc = self._model = self._hooks = None


def _action_checksum(action) -> str:
    """Checksum of the (32,8) action chunk, rounded to tolerate compile/kernel FP noise."""
    import numpy as np
    a = np.round(np.asarray(action, dtype="float64"), 3)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]
