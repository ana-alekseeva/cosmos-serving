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

    Reasoner served by vLLM (Qwen3-VL path), Generator/full policy by vLLM-Omni. The
    endpoint returns server-side per-stage timers (CUDA events); the client adds transport.
    Written from the serving contract — confirm every `# VERIFY` before trusting numbers.
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
        import time

        from policy.serving import submit_policy_request  # VERIFY: request/response schema
        t0 = time.perf_counter()
        resp = submit_policy_request(self.endpoint, self.model, req, self.config)
        transport_ms = (time.perf_counter() - t0) * 1e3 - resp["server_ms"]
        t = resp["latency_ms"]              # VERIFY: server returns this per-stage block
        steps = resp.get("denoising_step_ms", [])
        server = resp["server_ms"]
        gate = resp.get("quality_gate", "passed" if not self.config.lossy else "n/a")
        return LatencyRecord(
            request_id=req.request_id, task=req.task, episode_id=req.episode_id,
            preprocess_ms=t["preprocess"], h2d_ms=t["h2d"], reasoner_ms=t["reasoner"],
            generator_prepare_ms=t["generator_prepare"], denoising_ms=t["denoising"],
            denoising_step_ms=steps, postprocess_ms=t["postprocess"], d2h_ms=t["d2h"],
            server_ms=server, transport_ms=max(0.0, transport_ms),
            first_action_ms=resp.get("first_action_ms", server + max(0.0, transport_ms)),
            total_chunk_ms=server + max(0.0, transport_ms),
            peak_memory_mb=resp.get("peak_memory_mb", 0.0),
            output_checksum=resp.get("output_checksum", ""), quality_gate=gate,
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
