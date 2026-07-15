"""Policy inference pipeline — REAL backend + engine factory. One request = one DROID observation -> one 32x8 action chunk."""
from __future__ import annotations

import warnings

from policy import compat
from policy.configs import Config
from policy.dataset import DroidRequest
from policy.measure import LatencyRecord
from policy.mock.engine import MockPolicyEngine
from policy.pytorch_engine import PyTorchPolicyEngine


class VLLMPolicyEngine:
    """Real backend: replay each DROID request against the deployed policy endpoint.

    ONE vllm-omni engine serves both towers; "Reasoner/Generator" name phases within a request.
    """

    backend = "vllm"

    def __init__(self, config: Config, *, model: str | None = None,
                 endpoint: str | None = None, **_):
        self.config = config
        self.model = model or "nvidia/Cosmos3-Nano-Policy-DROID"
        self.endpoint = endpoint
        self._server = None

    def prepare(self) -> None:
        import os

        compat.validate(self.config, "vllm")
        for pair, why in compat.conflicts(self.config):
            names = " + ".join(compat.TECHNIQUES[k][0] for k in pair)
            warnings.warn(f"{self.config.cid}: {names} — {why}", stacklevel=2)
        if self.endpoint:                   # externally-deployed endpoint: nothing to launch
            return
        # Per-config SUBDIR so traces are attributable; safe to mutate os.environ (per-config subprocess).
        base = os.environ.get("VLLM_TORCH_PROFILER_DIR")
        if base:
            os.environ["VLLM_TORCH_PROFILER_DIR"] = os.path.join(base, self.config.cid)
            os.makedirs(os.environ["VLLM_TORCH_PROFILER_DIR"], exist_ok=True)
        from policy.serving import start_policy_server
        self._server = start_policy_server(self.model, self.config)
        self.endpoint = self._server.base_url

    def run_request(self, req: DroidRequest) -> LatencyRecord:
        import hashlib
        import json
        import time

        from policy.serving import submit_policy_request
        t0 = time.perf_counter()
        resp = submit_policy_request(self.endpoint, self.model, req, self.config)
        total_ms = (time.perf_counter() - t0) * 1e3
        # inference_time_s is the authoritative server total; total_chunk_ms stays client-measured.
        server = float(resp.get("inference_time_s", total_ms / 1e3)) * 1e3
        # stage_durations values already in ms; fold *_gen_ms into denoising_ms (one omni stage, both towers).
        stages = resp.get("stage_durations") or {}
        denoising_ms = sum(float(v) for k, v in stages.items()
                           if str(k).endswith("_gen_ms")) if isinstance(stages, dict) else 0.0
        # action is a VideoAction object: {data, shape, dtype, raw_action_dim, ...}
        action = resp.get("action", resp.get("actions"))
        action_data = action.get("data") if isinstance(action, dict) else action
        checksum = hashlib.sha256(json.dumps(action_data).encode()).hexdigest()[:16]
        gate = resp.get("quality_gate", "passed" if not self.config.lossy else "n/a")
        return LatencyRecord(
            request_id=req.request_id, task=req.task, episode_id=req.episode_id,
            preprocess_ms=0.0, h2d_ms=0.0, reasoner_ms=0.0,
            generator_prepare_ms=0.0, denoising_ms=denoising_ms,
            denoising_step_ms=resp.get("denoising_step_ms", []),
            postprocess_ms=0.0, d2h_ms=0.0,
            server_ms=server, transport_ms=max(0.0, total_ms - server),
            first_action_ms=resp.get("first_action_ms", total_ms),
            total_chunk_ms=total_ms,
            peak_memory_mb=float(resp.get("peak_memory_mb", 0.0) or 0.0),
            output_checksum=resp.get("output_checksum", checksum), quality_gate=gate,
        )

    def capture_profile(self, req: DroidRequest) -> None:
        """One EXTRA profiled request AFTER the measured pass, so profiler overhead never
        contaminates the records. No-ops unless VLLM_TORCH_PROFILER_DIR is set."""
        import os

        if not os.environ.get("VLLM_TORCH_PROFILER_DIR") or self.endpoint is None:
            return
        from policy.serving import start_profile, stop_profile
        start_profile(self.endpoint)
        try:
            self.run_request(req)                     # traced request; record discarded
        finally:
            stop_profile(self.endpoint)

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None


BACKENDS = ("mock", "pytorch", "vllm")


def make_engine(backend: str, config: Config, *, model: str | None = None,
                endpoint: str | None = None, checkpoint_dir: str | None = None):
    """Build a policy engine. `pytorch` = native reference (waterfall, Job 1);
    `vllm` = vLLM/vLLM-Omni production (Job 2); `mock` = modeled, no GPU."""
    if backend == "mock":
        return MockPolicyEngine(config, model=model)
    if backend == "pytorch":
        return PyTorchPolicyEngine(config, model=model, checkpoint_dir=checkpoint_dir)
    if backend == "vllm":
        return VLLMPolicyEngine(config, model=model, endpoint=endpoint)
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
