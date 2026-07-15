"""Real-backend serving contract for the policy pipeline (specification_revised.txt §4
Job 2, §9) — STOCK vLLM-Omni (>=0.22 registers Cosmos3OmniDiffusersPipeline, the
checkpoint's declared class; pins in deploy/versions.env, feasibility-verified on H100).

Topology note: ONE engine serves BOTH towers. The Qwen3-VL Reasoner and the diffusion
action expert live inside the single Cosmos3 pipeline (one `vllm-omni serve` process, one
endpoint) — "Reasoner stage" / "Generator stage" name phases WITHIN a request, not separate
deployments. Contract verified against the vllm-omni 0.24.0 source (2026-07-15):

  * Serve: `vllm-omni serve <model> [engine flags]` — technique flags are ENGINE args.
  * Request: `POST /v1/videos/sync` multipart form — the conditioning camera frame is the
    `input_reference` file; the Generator recipe (num_inference_steps/guidance_scale/
    flow_shift/seed) are per-REQUEST form fields; action plumbing goes in the
    `extra_params` JSON (action_mode="policy", domain_name="droid_lerobot",
    action_chunk_size, robot_obs) merged into the pipeline's extra_args.
  * Response: top-level `action` field, [chunk, dim] = [32, 8] for DROID.
  * /v1/chat/completions only reaches the language tower (it chats; no actions).

Remaining `# VERIFY` items are named on-box checks, not guesses.
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
    """Config -> vLLM-Omni ENGINE flags (serve-time). The Generator sampling recipe is NOT
    here — it is per-request form fields (request_form_fields below), per the 0.24 API."""
    flags = config.stage_flags
    args: list[str] = ["--max-num-seqs", "1"]        # batch size 1 (§6); model is positional

    # Attention backend for the DIFFUSION tower, per-role dot-notation (vllm-omni docs form).
    # Force it explicitly; do NOT silently fall back (§9).
    if flags.get("attention") == "flash":
        args += ["--diffusion-attention-config.per_role.self.backend", "FLASH_ATTN"]
    else:
        args += ["--diffusion-attention-config.per_role.self.backend", "TORCH_SDPA"]  # VERIFY value

    # torch.compile / CUDA graphs, kept distinct so they are not double-counted (§9).
    # Verified 0.24 semantics: the DIFFUSION tower compiles iff --enforce-eager is ABSENT
    # (diffusion_model_runner regionally_compiles the transformer; no diffusion cudagraph
    # knob exists), while --compilation-config governs the AR tower (CompilationMode enum:
    # NONE/STOCK_TORCH_COMPILE/DYNAMO_TRACE_ONCE/VLLM_COMPILE + cudagraph_mode). So E2 =
    # compile both towers/no graphs; E3 adds CUDA graphs — AR tower only, by construction.
    if flags.get("cuda_graphs"):
        args += ["--compilation-config",
                 '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL_AND_PIECEWISE"}']
    elif flags.get("compile"):
        args += ["--compilation-config", '{"mode":"VLLM_COMPILE","cudagraph_mode":"NONE"}']
    else:
        args += ["--enforce-eager"]

    # Reasoner conditioning cache (E4): NOT a stock flag — within-request K/V caching is
    # built in; the CROSS-request conditioning cache is our patch (worth it for THIS workload:
    # the replay set cycles 50 observations x10, so consecutive requests do share instructions).
    # Until the patch exists, E4 measures identical to E3 on this backend — flagged, not silent.
    if flags.get("reasoner_cache"):
        warnings.warn(f"{config.cid}: cross-request conditioning cache requires our "
                      "vllm-omni patch; serving stock (E4 == E3 on this backend)", stacklevel=2)

    # Cache-DiT (lossy) — real backend (vllm_omni/diffusion/cache/cache_dit_backend.py).
    if flags.get("cache_dit"):
        args += ["--cache-backend", "cache_dit"]     # VERIFY flag spelling vs DIFFUSION_CACHE_BACKEND env

    # Dynamic FP8 (lossy).
    if flags.get("quantization") == "fp8":
        args += ["--quantization", "fp8"]            # VERIFY applies to the diffusion tower on 0.24

    # GPU-op traces. VERIFIED against vllm 0.19.1 + vllm-omni 0.20.0 source (2026-07-15): the
    # VLLM_TORCH_PROFILER_DIR env var was REMOVED from vLLM — profiling is enabled via the
    # --profiler-config engine flag (vllm/config/profiler.py), and the /start_profile +
    # /stop_profile routes are only MOUNTED when a stage's engine args carry a profiler_config
    # with profiler set (vllm_omni api_server._should_enable_profiler_endpoints). We keep the
    # env var as OUR job-level knob (run_job.sh sets it; pipeline.py points it at a per-config
    # subdir before launch) and translate it to the flag here.
    profiler_dir = os.environ.get("VLLM_TORCH_PROFILER_DIR")
    if profiler_dir:
        args += ["--profiler-config",
                 json.dumps({"profiler": "torch", "torch_profiler_dir": profiler_dir})]

    # Multi-GPU (jobs/job2b, §5.3.3) — vllm-omni's names (Omni ctor: cfg_parallel_size,
    # ulysses_degree). Driven by env so a job sets it without touching the config matrix.
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
    """Per-REQUEST Generator recipe (steps=4, guidance=3, shift=5) + fixed seed — identical
    across every rung so the technique (not a changed schedule) explains the delta (§9/§10).
    These are /v1/videos form fields on 0.24, NOT serve flags. cfg_mode has no form field —
    CFG Null semantics live in the checkpoint's pipeline config (VERIFY on-box)."""
    s = GENERATOR_SAMPLING
    return {
        "num_inference_steps": str(s.steps),
        "guidance_scale": str(s.guidance),
        "flow_shift": str(s.shift),
        "seed": str(seed),
    }


def start_profile(endpoint: str) -> None:
    """Start vLLM's server-side torch profiler (traces -> the --profiler-config
    torch_profiler_dir; engine_args() sets it from VLLM_TORCH_PROFILER_DIR). The route only
    exists when the server was launched with a profiler_config (verified: vllm-omni 0.20.0
    mounts /start_profile conditionally); callers treat an HTTP error as "no trace", not fatal."""
    urllib.request.urlopen(urllib.request.Request(endpoint.rstrip("/") + "/start_profile",
                                                  method="POST"), timeout=30)


def stop_profile(endpoint: str) -> None:
    """Stop vLLM's profiler; the server flushes the Chrome trace to VLLM_TORCH_PROFILER_DIR."""
    urllib.request.urlopen(urllib.request.Request(endpoint.rstrip("/") + "/stop_profile",
                                                  method="POST"), timeout=120)


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


# Serve entrypoint — verified against vllm-omni 0.24.0 (cli/main.py): the `vllm-omni` binary
# WITHOUT `--omni` silently dispatches to VANILLA vllm — AR language tower only (chat routes
# work, /v1/videos 404s, no diffusion/action stage; feasibility S5 caught this). `--omni` is
# what selects omni's multi-stage server, so it is part of the command, not an option.
SERVE_CMD = ("vllm-omni", "serve")
OMNI_FLAG = "--omni"
HEALTH_ROUTE = "/health"                         # verified (feasibility S3)
# ASYNC videos route: the sync variant returns raw mp4 bytes and DISCARDS the action; the
# async job record carries action + stage_durations + inference_time_s + peak_memory_mb.
INFER_ROUTE = "/v1/videos"
POLL_INTERVAL_S = 0.025                          # bounded client-side noise on total_chunk_ms


def start_policy_server(model: str, config: Config, *, host: str = "127.0.0.1",
                        port: int = 8000, ready_timeout_s: float = 900.0) -> ServerHandle:
    """Launch vLLM-Omni with `config`'s engine flags and block until /health is 200.

    VERIFY on-box: the serve entrypoint (SERVE_CMD), that every engine_args() flag is accepted,
    the static-shape / bucketing config the CUDA-graph rungs need (§9), and the readiness route.
    """
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
    """Client-side replica of the server's compose_robolab_views concat_view: wrist on top,
    the two DROID exterior views halved side-by-side below. Older 2-view captures (no
    exterior_2 in the .npz) reuse the single exterior for both halves."""
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

    robot_obs schema VERIFIED against the 0.24.0 pipeline (_build_robolab_policy_inputs):
    string key 'prompt' (required), 'observation/joint_position' [T,7] and
    'observation/gripper_position' [T,1] (server inverts to 1-x; use_state defaults true so
    the last row conditions the action expert). Camera keys are deliberately ABSENT from
    robot_obs (base64 wouldn't parse — the server wants uint8 arrays): extract_robolab_image
    then falls back to the request's multi_modal_data.image = our input_reference file, which
    we send as the composed concat_view PNG (wrist top, both DROID exteriors below) — binary,
    not JSON."""
    from policy.capture import load_capture

    obs = load_capture(req.capture_ref)             # real DROID observation (exterior/wrist/proprio)
    proprio = [float(x) for x in obs["proprio"]]    # 8-D: joint(7) + gripper(1)
    instruction = str(obs["instruction"])
    # FastAPI drops an EMPTY multipart form value entirely (parsed as None -> 400 "prompt
    # Field required"; reproduced against the same Form signature). Some DROID episodes have
    # no language annotation. The form prompt is only the video API's required field — the
    # actual conditioning text is robot_obs["prompt"] (ai_caption), which keeps the true "".
    fields = {
        "model": model,
        "prompt": instruction if instruction.strip() else " ",
        **request_form_fields(req.seed),            # steps/guidance/shift/seed (fixed, §10)
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
    /v1/videos, then poll GET /v1/videos/{id} until completed).

    The completed job record (verified against the 0.24.0 source) carries `action`
    ([32, 8] for DROID), `stage_durations` (SERVER-side per-stage timing), `inference_time_s`
    and `peak_memory_mb` — the caller maps those into the LatencyRecord. Polling adds at most
    POLL_INTERVAL_S of client-side noise to wall time; `inference_time_s` is authoritative
    for the server. Fail loudly if no action comes back."""
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
