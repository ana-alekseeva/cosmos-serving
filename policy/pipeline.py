"""Policy inference pipeline — REAL backend + engine factory (specification_revised.txt §6).

One request = one DROID observation -> one 32x8 action chunk.

  - VLLMPolicyEngine: real path — sends each replay request to the deployed vLLM (Reasoner)
    / vLLM-Omni (Generator/full policy) endpoint, reads server-side per-stage timers, adds
    client-side transport. Written against the serving contract; every `# VERIFY` confirmed
    on-box.

The MOCK backend (modeled per-stage latency, no GPU) lives in policy/mock/engine.py;
`make_engine` wires it in for `--backend mock`.
"""
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

    ONE vllm-omni engine serves both towers (the Qwen3-VL Reasoner and the diffusion action
    expert live inside the stock Cosmos3OmniDiffusersPipeline) — "Reasoner/Generator" name
    phases within a request, not separate deployments. Timing is client-measured
    total_chunk_ms; per-stage attribution comes from profiler traces (capture_profile).
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

        compat.validate(self.config, "vllm")                 # §5.3.3: vLLM-Omni technique set
        for pair, why in compat.conflicts(self.config):      # §9: warn on non-composing pairs
            names = " + ".join(compat.TECHNIQUES[k][0] for k in pair)
            warnings.warn(f"{self.config.cid}: {names} — {why}", stacklevel=2)
        if self.endpoint:                   # externally-deployed endpoint: nothing to launch
            return
        # GPU-op traces (the vLLM analogue of the native path's Perfetto traces): serving.py
        # turns VLLM_TORCH_PROFILER_DIR into the --profiler-config engine flag (the env var
        # itself was removed from vllm 0.19.1); traces flush on /stop_profile (capture_profile
        # below). Point it at a per-config SUBDIR so trace_E0 vs trace_E6 are attributable —
        # safe to mutate os.environ: each config runs in its own subprocess (§4), and
        # engine_args() reads it at launch time.
        base = os.environ.get("VLLM_TORCH_PROFILER_DIR")
        if base:
            os.environ["VLLM_TORCH_PROFILER_DIR"] = os.path.join(base, self.config.cid)
            os.makedirs(os.environ["VLLM_TORCH_PROFILER_DIR"], exist_ok=True)
        # VERIFY: launch vLLM-Omni with this config's engine flags (bench serving contract).
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
        # The completed async job record carries SERVER-side timing: inference_time_s (the
        # authoritative server total) and stage_durations (per-omni-stage seconds). Map the
        # omni stages onto our §7 fields heuristically (AR/reasoner-ish -> reasoner_ms,
        # diffusion-ish -> denoising_ms); finer attribution comes from profiler traces
        # (capture_profile). total_chunk_ms stays client-measured (includes <=25ms poll noise).
        server = float(resp.get("inference_time_s", total_ms / 1e3)) * 1e3
        stages = resp.get("stage_durations") or {}
        reasoner_ms = denoising_ms = 0.0
        for name, seconds in (stages.items() if isinstance(stages, dict) else []):
            key = str(name).lower()
            if "diffusion" in key:
                denoising_ms += float(seconds) * 1e3
            elif any(k in key for k in ("ar", "llm", "reason", "text")):
                reasoner_ms += float(seconds) * 1e3
        action = resp.get("action", resp.get("actions"))
        checksum = hashlib.sha256(json.dumps(action).encode()).hexdigest()[:16]
        gate = resp.get("quality_gate", "passed" if not self.config.lossy else "n/a")
        return LatencyRecord(
            request_id=req.request_id, task=req.task, episode_id=req.episode_id,
            preprocess_ms=0.0, h2d_ms=0.0, reasoner_ms=reasoner_ms,
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
        """One EXTRA request under vLLM's server-side torch profiler — the GPU-op trace (Perfetto
        dashboard) for this config. Called by the runner AFTER the measured pass, so profiler
        overhead never contaminates the latency records. No-ops unless the server was launched
        with VLLM_TORCH_PROFILER_DIR set (run_job.sh sets it for BACKEND=vllm)."""
        import os

        if not os.environ.get("VLLM_TORCH_PROFILER_DIR") or self.endpoint is None:
            return
        from policy.serving import start_profile, stop_profile
        start_profile(self.endpoint)                  # VERIFY the /start_profile route on-box
        try:
            self.run_request(req)                     # traced request; record discarded
        finally:
            stop_profile(self.endpoint)               # server flushes the Chrome trace

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None


BACKENDS = ("mock", "pytorch", "vllm")


def make_engine(backend: str, config: Config, *, model: str | None = None,
                endpoint: str | None = None, checkpoint_dir: str | None = None):
    """Build a policy engine. `pytorch` = native §5.3.1 reference (waterfall, Job 1);
    `vllm` = vLLM/vLLM-Omni production (§5.3.2/§5.3.3, Job 2); `mock` = modeled, no GPU."""
    if backend == "mock":
        return MockPolicyEngine(config, model=model)
    if backend == "pytorch":
        return PyTorchPolicyEngine(config, model=model, checkpoint_dir=checkpoint_dir)
    if backend == "vllm":
        return VLLMPolicyEngine(config, model=model, endpoint=endpoint)
    raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
