"""Real-backend serving contract for the policy pipeline (Job 2) — STOCK vLLM-Omni.

ONE engine serves BOTH towers (Qwen3-VL Reasoner + diffusion action expert) inside a single
Cosmos3 pipeline. Contract verified against the vllm-omni 0.24.0 source (2026-07-15):
technique flags are serve-time ENGINE args; requests go to `POST /v1/videos` multipart
(input_reference frame + per-request recipe fields + extra_params action plumbing); the
response carries a top-level `action` [32, 8] for DROID.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass

from policy.configs import GENERATOR_SAMPLING, REASONER_SAMPLING, Config

DEFAULT_MODEL = "nvidia/Cosmos3-Nano-Policy-DROID"


def reasoner_sampling_params() -> dict:
    """vLLM SamplingParams for the Reasoner (Qwen3-VL) conditioning pass (deterministic decode)."""
    s = REASONER_SAMPLING
    return {
        "max_tokens": s.max_tokens,
        "temperature": s.temperature,
        "top_p": s.top_p,
        "top_k": s.top_k,
        "repetition_penalty": s.repetition_penalty,
    }


def engine_args(config: Config) -> list[str]:
    """Config -> vLLM-Omni ENGINE flags (serve-time). The Generator recipe is per-request (request_form_fields)."""
    flags = config.stage_flags
    args: list[str] = ["--max-num-seqs", "1"]        # batch size 1; model is positional

    # Diffusion-tower attention backend; force explicitly, do NOT silently fall back.
    if flags.get("attention") == "flash":
        args += ["--diffusion-attention-config.per_role.self.backend", "FLASH_ATTN"]
    else:
        args += ["--diffusion-attention-config.per_role.self.backend", "TORCH_SDPA"]  # VERIFY value

    # 0.24 semantics: diffusion tower compiles iff --enforce-eager ABSENT; --compilation-config
    # governs the AR tower only. So E3's CUDA graphs are AR-tower-only by construction.
    if flags.get("cuda_graphs"):
        args += ["--compilation-config",
                 '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}']
    elif flags.get("compile"):
        args += ["--compilation-config", '{"mode":"VLLM_COMPILE","cudagraph_mode":"NONE"}']
    else:
        args += ["--enforce-eager"]

    # Reasoner conditioning cache (E4): cross-request cache is our patch; stock serves E4 == E3.
    if flags.get("reasoner_cache"):
        warnings.warn(f"{config.cid}: cross-request conditioning cache requires our "
                      "vllm-omni patch; serving stock (E4 == E3 on this backend)", stacklevel=2)

    # Cache-DiT (lossy).
    if flags.get("cache_dit"):
        args += ["--cache-backend", "cache_dit"]     # VERIFY flag spelling vs DIFFUSION_CACHE_BACKEND env

    # Dynamic FP8 (lossy).
    if flags.get("quantization") == "fp8":
        args += ["--quantization", "fp8"]            # VERIFY applies to the diffusion tower on 0.24

    # VLLM_TORCH_PROFILER_DIR was REMOVED from vLLM; translate our job-level env knob to --profiler-config.
    profiler_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if profiler_dir:
        args += ["--profiler-config",
                 json.dumps({"profiler": "torch", "torch_profiler_dir": profiler_dir})]

    # Multi-GPU (jobs/job2b), driven by env so a job sets it without touching the config matrix.
    tp = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
    if tp > 1:
        parallel = os.environ.get("PARALLEL", "cfg")                  # cfg | ulysses
        if parallel == "cfg":
            args += ["--cfg-parallel-size", str(tp)]                  # VERIFY CLI form of cfg_parallel_size
        elif parallel == "ulysses":
            args += ["--ulysses-degree", str(tp)]                     # VERIFY CLI form of ulysses_degree
        else:
            args += ["--tensor-parallel-size", str(tp)]
    return args


def request_form_fields(seed: int) -> dict[str, str]:
    """Per-REQUEST Generator recipe + fixed seed (identical across rungs); /v1/videos form fields, NOT serve flags."""
    s = GENERATOR_SAMPLING
    return {
        "num_inference_steps": str(s.steps),
        "guidance_scale": str(s.guidance),
        "flow_shift": str(s.shift),
        "seed": str(seed),
    }


def start_profile(endpoint: str) -> None:
    """Start vLLM's server-side torch profiler; route only exists if launched with a profiler_config."""
    urllib.request.urlopen(urllib.request.Request(endpoint.rstrip("/") + "/start_profile",
                                                  method="POST"), timeout=30)


def stop_profile(endpoint: str) -> None:
    """Stop vLLM's profiler; the server flushes the Chrome trace to VLLM_TORCH_PROFILER_DIR."""
    urllib.request.urlopen(urllib.request.Request(endpoint.rstrip("/") + "/stop_profile",
                                                  method="POST"), timeout=120)


def sdpa_attention_snippet() -> str:
    """The forced-Flash SDPA pattern the eager path uses (documented for reference)."""
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
            proc.wait(timeout=30)               # let it drain the CUDA context
        except subprocess.TimeoutExpired:
            proc.kill()                          # force if it will not exit
        except Exception:
            pass


# `vllm-omni` WITHOUT `--omni` silently dispatches to vanilla vllm (no diffusion/action stage).
SERVE_CMD = ("vllm-omni", "serve")
OMNI_FLAG = "--omni"
HEALTH_ROUTE = "/health"
# ASYNC videos route: the /sync variant returns raw mp4 bytes and DISCARDS the action.
INFER_ROUTE = "/v1/videos"
POLL_INTERVAL_S = 0.025                          # bounded client-side noise on total_chunk_ms


def start_policy_server(model: str, config: Config, *, host: str = "127.0.0.1",
                        port: int = 8000, ready_timeout_s: float = 900.0) -> ServerHandle:
    """Launch vLLM-Omni with `config`'s engine flags and block until /health is 200."""
    cmd = [*SERVE_CMD, model, OMNI_FLAG, *engine_args(config), "--host", host, "--port", str(port)]
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


def _png(arr) -> bytes:
    """uint8 HWC array -> PNG bytes (PIL ships with vllm's dep tree)."""
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _concat_view(wrist, exterior, exterior_2=None):
    """Client-side replica of the server's concat_view: wrist on top, two exterior views halved below."""
    import numpy as np
    from PIL import Image

    h, w = wrist.shape[0] // 2, wrist.shape[1] // 2

    def _half(view):
        return np.asarray(Image.fromarray(view).resize((w, h)), dtype=np.uint8)

    half = _half(exterior)
    half2 = _half(exterior_2) if exterior_2 is not None else half
    return np.concatenate([wrist, np.concatenate([half, half2], axis=1)], axis=0)


def build_request_parts(req, model: str) -> tuple[dict[str, str], bytes]:
    """One DROID observation -> (/v1/videos form fields, input_reference PNG bytes).

    Camera keys are deliberately ABSENT from robot_obs (the server wants uint8 arrays, not
    base64); the frame rides in as the multipart input_reference file instead.
    """
    from policy.capture import load_capture

    obs = load_capture(req.capture_ref)             # real DROID observation (exterior/wrist/proprio)
    proprio = [float(x) for x in obs["proprio"]]    # 8-D: joint(7) + gripper(1)
    instruction = str(obs["instruction"])
    # FastAPI drops an EMPTY multipart value (-> 400 "prompt Field required"); send " " here.
    # True conditioning text is robot_obs["prompt"], which keeps the real "".
    fields = {
        "model": model,
        "prompt": instruction if instruction.strip() else " ",
        **request_form_fields(req.seed),            # steps/guidance/shift/seed (fixed)
        # server-enforced: num_frames == action_chunk_size (or +1) for action requests
        "num_frames": "32",
        "extra_params": json.dumps({
            "action_mode": "policy",
            "domain_name": "droid_lerobot",         # -> domain_id 8 (EMBODIMENT_TO_DOMAIN_ID)
            "action_chunk_size": 32,
            "raw_action_dim": 8,                    # DROID joint_pos space: joint(7)+gripper(1)
            "robot_obs": {
                "prompt": str(obs["instruction"]),
                "observation/joint_position": [proprio[:7]],
                "observation/gripper_position": [[proprio[7]]],
            },
        }),
    }
    return fields, _png(_concat_view(obs["wrist"], obs["exterior"], obs.get("exterior_2")))


def _multipart(fields: dict[str, str], file_field: str, filename: str,
               file_bytes: bytes) -> tuple[bytes, str]:
    import uuid
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
             f"filename=\"{filename}\"\r\nContent-Type: image/png\r\n\r\n").encode()
    body += file_bytes + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def submit_policy_request(endpoint: str, model: str, req, config: Config,
                          *, timeout_s: float = 600.0) -> dict:
    """One DROID observation -> action chunk via the ASYNC videos API (multipart POST
    /v1/videos, then poll GET /v1/videos/{id} until completed). Fails loudly if no action."""
    fields, frame = build_request_parts(req, model)
    body, content_type = _multipart(fields, "input_reference", "exterior.png", frame)
    url = endpoint.rstrip("/") + INFER_ROUTE
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": content_type, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            ref = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"policy endpoint {url} returned {e.code}: {e.read()[:800]!r}") from e

    job_url = f"{url}/{ref['id']}"
    deadline = time.monotonic() + timeout_s
    job: dict = ref
    while time.monotonic() < deadline:
        with urllib.request.urlopen(job_url, timeout=30) as resp:
            job = json.loads(resp.read().decode("utf-8"))
        status = str(job.get("status", "")).lower()
        if "completed" in status:
            break
        if "failed" in status:
            raise RuntimeError(f"policy job {ref['id']} failed: {job.get('error')}")
        time.sleep(POLL_INTERVAL_S)
    else:
        raise TimeoutError(f"policy job {ref['id']} not completed within {timeout_s:.0f}s")

    action = job.get("action", job.get("actions"))
    if action is None:
        raise KeyError(f"completed policy job has no action; got keys {sorted(job)}")
    return job
