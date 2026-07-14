"""Native PyTorch reference backend (Cosmos 3 report §5.3.1) via cosmos_framework.

Runs Cosmos3-Nano-Policy-DROID through cosmos_framework.inference.inference.OmniInference — the
same path `action_policy_server_robolab.py` serves it with. The native inference is ONE call,
`model.generate_samples_from_batch(batch, guidance, seed, num_steps, shift)`, returning
`samples["action"]` of shape [T, D] (D=8 joint_pos). It is NOT a decomposed reasoner/denoiser
API, which has two consequences:

  * The §5.3.1 optimizations map to OmniSetup flags applied at LOAD time — torch.compile ->
    `use_torch_compile`, CUDA graphs -> `use_cuda_graphs`. We do NOT hand-roll torch.compile /
    CUDA-graph capture. (No FP8 field in the setup layer — FP8 is vLLM-Omni-only, §5.3.3.)
  * Per-stage waterfall timing (reasoner vs denoising) is NOT separable from one black-box
    call. This engine times preprocess / model-compute / postprocess and reports the model
    compute as a single number; splitting it into reasoner_ms + denoising_ms needs timers
    INSIDE cosmos_framework (instrument generate_samples_from_batch — a cosmos-side change).

FP8 + Cache-DiT are vLLM-Omni features (§5.3.3); those configs route to the vllm backend
(policy/compat.resolve_backend), never here.

Requires cosmos_framework on the box (github.com/NVIDIA/cosmos-framework, the policy-server
extras). Every cosmos_framework symbol below is a `# VERIFY` against your installed version.
"""
from __future__ import annotations

import hashlib

from policy import compat
from policy.configs import ACTION_CHUNK, GENERATOR_SAMPLING, Config
from policy.dataset import DroidRequest
from policy.measure import LatencyRecord

_ACTION_T, _ACTION_D = ACTION_CHUNK        # 32 timesteps x 8 DoF (§1)


def _make_setup(checkpoint_path: str, *, compile_: bool, cuda_graphs: bool):
    """Build the cosmos_framework OmniInference setup for this config's §5.3.1 flags.

    VERIFY the OmniSetupOverrides field names + build_setup() call against your version —
    action_policy_server_robolab.py builds it via `OmniSetupOverrides.model_validate(...)`
    then `build_setup(...)`. Attention (flash) is the model default; the reasoner-conditioning
    cache (R4/E4) maps to a cosmos flag TBD (VERIFY)."""
    from cosmos_framework.inference.inference import (  # VERIFY import path
        OmniSetupOverrides,
        build_setup,
    )
    overrides = OmniSetupOverrides.model_validate({
        "checkpoint_path": checkpoint_path,
        "use_torch_compile": compile_,       # §5.3.1 torch.compile (R2/G2/E2)
        "use_cuda_graphs": cuda_graphs,      # §5.3.1 CUDA-graph replay (R3/G3/E3)
    })
    return build_setup(overrides)


class PyTorchPolicyEngine:
    """Native eager-PyTorch reference path (§5.3.1) via cosmos_framework. Backend id: 'pytorch'."""

    backend = "pytorch"

    def __init__(self, config: Config, *, model: str | None = None,
                 checkpoint_dir: str | None = None, **_):
        compat.validate(config, "pytorch")           # refuse Cache-DiT/FP8 (§5.3.3)
        self.config = config
        # cosmos accepts a local dir or an HF id; prefer the staged checkpoint dir.
        self.checkpoint = checkpoint_dir or model or "nvidia/Cosmos3-Nano-Policy-DROID"
        self._model = None
        self._device = None

    def prepare(self) -> None:
        import torch
        from cosmos_framework.inference.inference import OmniInference   # VERIFY import path

        flags = self.config.stage_flags
        setup = _make_setup(
            self.checkpoint,
            compile_=bool(flags.get("compile") or flags.get("cuda_graphs")),  # graphs imply compile
            cuda_graphs=bool(flags.get("cuda_graphs")),
        )
        self._model = OmniInference.create(setup).model   # loads the 16B MoT policy
        self._device = torch.device("cuda")

    def run_request(self, req: DroidRequest) -> LatencyRecord:
        import torch

        gen = GENERATOR_SAMPLING
        dev = self._device
        torch.cuda.reset_peak_memory_stats(dev)

        def _ev():
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        pre, comp, post = _ev(), _ev(), _ev()
        with torch.inference_mode():
            pre[0].record()
            batch = self._build_batch(req)               # ai_caption + video + proprio + domain_id
            pre[1].record()

            comp[0].record()
            samples = self._model.generate_samples_from_batch(   # reasoner + diffusion + decode
                batch, guidance=gen.guidance, seed=[req.seed],
                num_steps=gen.steps, shift=gen.shift)            # VERIFY kwarg names
            comp[1].record()

            post[0].record()
            action = samples["action"][0][:_ACTION_T, :_ACTION_D]   # [32, 8]  VERIFY history trim
            action_cpu = action.detach().to("cpu")
            post[1].record()

        torch.cuda.synchronize(dev)
        preprocess_ms = pre[0].elapsed_time(pre[1])
        compute_ms = comp[0].elapsed_time(comp[1])       # reasoner+denoise combined (single call)
        postprocess_ms = post[0].elapsed_time(post[1])
        server_ms = preprocess_ms + compute_ms + postprocess_ms
        peak_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)

        return LatencyRecord(
            request_id=req.request_id, task=req.task, episode_id=req.episode_id,
            preprocess_ms=preprocess_ms, h2d_ms=0.0,
            # Single-call model: reasoner/denoise are not separable here. The whole model compute
            # is reported as denoising_ms; reasoner_ms / generator_prepare_ms need cosmos_framework
            # instrumentation (timers inside generate_samples_from_batch) for the §3 stage split.
            reasoner_ms=0.0, generator_prepare_ms=0.0,
            denoising_ms=compute_ms, denoising_step_ms=[],
            postprocess_ms=postprocess_ms, d2h_ms=0.0,
            server_ms=server_ms, transport_ms=0.0,       # in-process: no client/server transport
            first_action_ms=server_ms, total_chunk_ms=server_ms,
            peak_memory_mb=peak_mb,
            output_checksum=_action_checksum(action_cpu),
            quality_gate="n/a",                          # pytorch runs only lossless configs
        )

    def _build_batch(self, req: DroidRequest):
        """Build the cosmos_framework data_batch from the real DROID capture.

        Mirror RobolabPolicyService._build_sample (action_policy_server_robolab.py): a dict with
          "ai_caption": the language instruction,
          "video":      image tensor [3, T, H, W] (DROID exterior view, resized to the model's
                        480p 640x360),
          "action":     a placeholder chunk,
          "domain_id":  the DROID domain id,
        plus the proprioceptive joint-position state. VERIFY the exact keys / resolution / dtype."""
        from policy.capture import load_capture
        obs = load_capture(req.capture_ref)              # exterior/wrist/proprio/instruction
        raise NotImplementedError(
            "build the cosmos_framework data_batch from the DROID capture — mirror "
            "RobolabPolicyService._build_sample in action_policy_server_robolab.py "
            "(ai_caption=instruction, video=[3,T,H,W] @480p 640x360, proprio joint_pos, domain_id). "
            f"capture has keys {sorted(obs)}.")

    def close(self) -> None:
        self._model = None


def _action_checksum(action) -> str:
    """Checksum of the (32,8) action chunk, rounded to tolerate compile/kernel FP noise so
    lossless rungs (compile/graphs) match the eager baseline (§10)."""
    import numpy as np
    a = np.round(np.asarray(action, dtype="float64"), 3)
    return hashlib.sha1(a.tobytes()).hexdigest()[:16]
