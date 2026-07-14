"""Real-backend serving contract for the policy pipeline (specification_revised.txt §4
Job 2, §9). vLLM (Reasoner, Qwen3-VL path) + vLLM-Omni (Generator / full policy).

This is the on-GPU wiring the mock stands in for. It is written from the serving docs and
is NOT exercised by the mock tests — confirm every `# VERIFY` against your installed
vLLM / vLLM-Omni before trusting a number. Two responsibilities:

  1. Map a Config's `stage_flags` to engine launch args (attention backend, compile,
     CUDA graphs, conditioning cache, Cache-DiT, FP8).
  2. Launch the server and submit a DROID request, returning the §7 per-stage timing block.

§9 compatibility rules encoded here:
  - Flash attention via forced SDPA backend (fail, don't silently fall back).
  - torch.compile and CUDA graphs measured without double-counting.
  - Static/bucketed shapes required for CUDA-graph configs.
"""
from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from policy.configs import GENERATOR_SAMPLING, REASONER_SAMPLING, Config

DEFAULT_MODEL = "nvidia/Cosmos3-Nano-Policy-DROID"


def reasoner_sampling_params() -> dict:
    """vLLM SamplingParams for the Reasoner (Qwen3-VL) conditioning pass.

    The Reasoner is stock vLLM and conditioning-only — it does not generate standalone text
    (§2) — decoded deterministically so conditioning is reproducible (§10). These go on the
    conditioning request (constant across the whole replay set). VERIFY the SamplingParams
    field names against your vLLM version and the canonical max_tokens on-box."""
    s = REASONER_SAMPLING
    return {
        "max_tokens": s.max_tokens,
        "temperature": s.temperature,
        "top_p": s.top_p,
        "top_k": s.top_k,
        "repetition_penalty": s.repetition_penalty,
    }


def engine_args(config: Config) -> list[str]:
    """Config -> vLLM-Omni serve flags. VERIFY every flag name against the serve CLI."""
    flags = config.stage_flags
    args: list[str] = ["--model", DEFAULT_MODEL, "--max-num-seqs", "1"]  # batch size 1 (§6)

    # Attention backend. Force it explicitly; do NOT silently fall back (§9). The eager path
    # uses torch SDPA with a forced backend (see sdpa_attention() below).
    if flags.get("attention") == "flash":
        args += ["--attention-backend", "FLASH_ATTN"]   # VERIFY: flag + accepted value
    else:
        args += ["--attention-backend", "TORCH_SDPA"]   # forced math SDPA baseline (§9 compare)

    # torch.compile / CUDA graphs. Kept distinct so they are not double-counted (§9):
    #   compile without graphs  vs  compile with graphs (reduce-overhead).
    if flags.get("cuda_graphs"):
        # If reduce-overhead auto-enables graph replay, treat as the combined config (§9).
        args += ["--compilation-config", '{"mode":"reduce-overhead"}']  # VERIFY schema
    elif flags.get("compile"):
        args += ["--compilation-config", '{"mode":"default"}']          # compile, no graphs
    else:
        args += ["--enforce-eager"]

    # Reasoner conditioning cache (P3/E4): compute conditioning once/observation (§3).
    if flags.get("reasoner_cache"):
        args += ["--policy-conditioning-cache", "true"]                 # VERIFY flag

    # Cache-DiT (lossy).
    if flags.get("cache_dit"):
        args += ["--cache-backend", "cache_dit"]                        # VERIFY flag

    # Dynamic FP8 (lossy).
    if flags.get("quantization") == "fp8":
        args += ["--quantization", "fp8"]                               # VERIFY flag

    # Generator sampling recipe — model-level, identical across every rung so the technique
    # (not a changed schedule) explains the delta. steps=4, guidance=3, shift=5, CFG Null.
    s = GENERATOR_SAMPLING
    args += ["--num-inference-steps", str(s.steps)]                     # VERIFY flag name
    args += ["--guidance-scale", str(s.guidance)]                      # VERIFY flag name
    args += ["--flow-shift", str(s.shift)]                             # VERIFY flag name
    args += ["--cfg-mode", s.cfg_mode]                                 # VERIFY flag + accepted value
    return args


def sdpa_attention_snippet() -> str:
    """The §9 forced-Flash SDPA pattern the eager path uses (documented for reference)."""
    return (
        "from torch.nn.attention import SDPBackend, sdpa_kernel\n"
        "import torch.nn.functional as F\n"
        "with sdpa_kernel(SDPBackend.FLASH_ATTENTION):   # fail, don't fall back (§9)\n"
        "    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,\n"
        "                                         dropout_p=0.0, is_causal=False)\n"
    )


@dataclass
class ServerHandle:
    base_url: str
    _proc: object = None

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        try:
            proc.terminate()
            proc.wait(timeout=30)               # let it drain the CUDA context (§4)
        except subprocess.TimeoutExpired:
            proc.kill()                          # force if it will not exit
        except Exception:
            pass


# Serve entrypoint (# VERIFY against your install: `vllm serve --omni` vs a dedicated
# vllm-omni CLI). engine_args() already carries --model and the per-config technique flags.
SERVE_CMD = ("vllm", "serve", "--omni")
HEALTH_ROUTE = "/health"                         # VERIFY readiness route
INFER_ROUTE = "/v1/policy/infer"                 # VERIFY inference route


def start_policy_server(model: str, config: Config, *, host: str = "127.0.0.1",
                        port: int = 8000, ready_timeout_s: float = 900.0) -> ServerHandle:
    """Launch vLLM-Omni with `config`'s engine flags and block until /health is 200.

    VERIFY on-box: the serve entrypoint (SERVE_CMD), that every engine_args() flag is accepted,
    the static-shape / bucketing config the CUDA-graph rungs need (§9), and the readiness route.
    """
    cmd = [*SERVE_CMD, *engine_args(config), "--host", host, "--port", str(port)]
    proc = subprocess.Popen(cmd)                 # inherits stdout/stderr -> job logs
    base_url = f"http://{host}:{port}"
    health = base_url + HEALTH_ROUTE
    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:              # server died during startup
            raise RuntimeError(
                f"policy server exited early (code {proc.returncode}). cmd: {' '.join(cmd)}")
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    return ServerHandle(base_url=base_url, _proc=proc)
        except (urllib.error.URLError, OSError):
            pass                                 # not up yet
        time.sleep(3)
    proc.terminate()
    raise TimeoutError(f"policy server not ready within {ready_timeout_s:.0f}s at {health}")


def _encode_image(arr) -> dict:
    """Compact on-wire image: base64 of the raw uint8 bytes + shape/dtype (NOT a nested int
    list — a 180x320x3 frame is ~170k ints in JSON). VERIFY the endpoint expects this shape."""
    return {"b64": base64.b64encode(arr.tobytes()).decode("ascii"),
            "shape": [int(x) for x in arr.shape], "dtype": str(arr.dtype)}


def build_request_payload(req, model: str) -> dict:
    """Build the on-wire payload for one DROID observation from its REAL captured tensors.

    Materializes the two camera views + 8-D proprio + instruction from `req.capture_ref`
    (a policy/capture.py .npz) and attaches the fixed Reasoner conditioning SamplingParams.
    The Generator recipe is baked into the server via engine_args(), so it is not repeated
    here. VERIFY the field names / image encoding against your deployed endpoint's schema."""
    from policy.capture import load_capture

    obs = load_capture(req.capture_ref)             # real DROID observation (exterior/wrist/proprio)
    return {
        "model": model,
        "images": {                                 # base64(uint8 bytes) + shape (VERIFY schema)
            "exterior": _encode_image(obs["exterior"]),
            "wrist": _encode_image(obs["wrist"]),
        },
        "proprio": [float(x) for x in obs["proprio"]],   # 8-D joint(7)+gripper(1)
        "prompt": obs["instruction"],
        "sampling": reasoner_sampling_params(),     # conditioning decode params (fixed, §10)
        "seed": req.seed,                           # fixed inference seed (reproducibility, §10)
    }


# §7 per-stage keys the server must return under `latency_ms` (CUDA-event timers).
_REQUIRED_STAGE_KEYS = ("preprocess", "h2d", "reasoner", "generator_prepare",
                        "denoising", "postprocess", "d2h")


def submit_policy_request(endpoint: str, model: str, req, config: Config) -> dict:
    """POST one DROID observation, return the §7 per-stage timing block.

    The payload (build_request_payload) carries the real captured observation + the Reasoner
    conditioning decode params; the Generator recipe is baked into the server via engine_args()
    — both fixed across the replay set.

    VERIFY on-box: the inference route (INFER_ROUTE) + payload schema, the action-chunk response
    shape (32x8), and that the server returns CUDA-event stage timers under `latency_ms`
    (preprocess/h2d/reasoner/generator_prepare/denoising/postprocess/d2h) plus `server_ms`.
    """
    payload = build_request_payload(req, model)
    data = json.dumps(payload).encode("utf-8")
    url = endpoint.rstrip("/") + INFER_ROUTE
    request = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:   # VERIFY server-side budget
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"policy endpoint {url} returned {e.code}: {e.read()[:500]!r}") from e

    # The client (policy/pipeline.py VLLMPolicyEngine.run_request) reads server_ms + a latency_ms
    # block with every stage key; fail loudly if the contract is not met rather than mis-timing.
    latency = body.get("latency_ms")
    if "server_ms" not in body or not isinstance(latency, dict):
        raise KeyError(f"policy response missing server_ms/latency_ms; got keys {sorted(body)}")
    missing = [k for k in _REQUIRED_STAGE_KEYS if k not in latency]
    if missing:
        raise KeyError(f"policy response latency_ms missing stages {missing}; got {sorted(latency)}")
    return body
