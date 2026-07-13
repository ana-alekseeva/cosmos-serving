"""Benchmark drivers for the real backend.

- Reasoner: NVIDIA **AIPerf** against the OpenAI-compatible endpoint, matching
  inference_benchmarks.md (fixed input/output tokens, concurrency, TTFT/latency).
- Generator: a timed generation request (wall-clock seconds/clip), matching the
  Generator methodology (diffusion has no tokens/s).

NOT yet run on-box. Every `# VERIFY` (AIPerf CLI flags, JSON schema, generation
endpoint/payload, multimodal inputs) must be confirmed against installed versions.
"""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from bench.workload import REASONER_VIDEO_DURATION_S, OperatingPoint


# Requests needed for a steady-state throughput/latency read scale with concurrency: send
# enough to cycle every concurrency slot several times over, not a flat count. At concurrency 1
# this collapses to the caller's floor (single-request OPs are unaffected).
STEADY_STATE_WAVES = 4


def run_aiperf(base_url: str, model: str, op: OperatingPoint,
               *, warmup: int = 2, request_count: int = 50) -> dict:
    """Reasoner latency via AIPerf at the OP's fixed shape + concurrency.

    `request_count` is a FLOOR; high-concurrency OPs are bumped to STEADY_STATE_WAVES ×
    concurrency so the measured window reflects steady state, not a half-filled ramp.
    """
    if shutil.which("aiperf") is None:
        raise RuntimeError("`aiperf` not found — install it on the GPU host (deploy/setup_gpu.sh).")
    n_requests = max(request_count, STEADY_STATE_WAVES * op.concurrency)
    out_dir = Path(tempfile.mkdtemp(prefix="aiperf-"))
    cmd = [
        "aiperf", "profile",
        "--model", model,
        "--url", base_url,
        "--endpoint-type", "chat",                                  # VERIFY endpoint type
        "--synthetic-input-tokens-mean", str(op.input_tokens),
        "--output-tokens-mean", str(op.output_tokens),
        "--extra-inputs", "ignore_eos:true",   # fixed output length -> comparable across variants
        "--concurrency", str(op.concurrency),
        "--request-count", str(n_requests),
        "--warmup-request-count", str(warmup),
        "--streaming",                          # required for a per-request TTFT measurement
        "--artifact-dir", str(out_dir),
    ]
    if op.modality == "image":
        w, h = _res_to_wh(op.clip_resolution)
        # AIPerf sends 1 synthetic image/request by default (--image-batch-size 1).
        cmd += ["--image-width-mean", str(w), "--image-height-mean", str(h)]
    elif op.modality == "video":
        w, h = _res_to_wh(op.clip_resolution)
        # Real synthetic video (NVIDIA "Video 1/2 FPS"): frames = fps x duration, so 2 FPS at the
        # same duration carries 2x the vision tokens. NB: do NOT pass --image-* here, or AIPerf
        # would also attach an image. Requires FFmpeg on the host.
        duration = max(1, round(op.clip_frames / op.video_fps)) if op.video_fps else REASONER_VIDEO_DURATION_S
        cmd += ["--video-fps", str(int(op.video_fps)), "--video-duration", str(duration),
                "--video-width", str(w), "--video-height", str(h),
                "--video-synth-type", "noise",   # content is semantically irrelevant; only shape matters
                # MP4/H.264, NOT the WebM/VP9 default: ffmpeg's synthetic WebM leaves the duration
                # header unset, so vLLM's frame sampler reads a sentinel and dies in
                # compute_frames_index_to_sample ("Number of samples ... must be non-negative").
                # MP4 writes an explicit duration in the moov atom.  # VERIFY vs vLLM video loader.
                "--video-format", "mp4", "--video-codec", "libx264"]
    subprocess.run(cmd, check=True)
    return _parse_aiperf(out_dir)


def _res_to_wh(res: str) -> tuple[int, int]:
    """Map an OP resolution tag to (width, height) for the synthetic frame. VERIFY dims."""
    return {"256p": (456, 256), "512px": (512, 512),
            "480p": (854, 480), "720p": (1280, 720)}.get(res, (256, 256))


def _parse_aiperf(out_dir: Path) -> dict:
    # the metrics file — NOT server_metrics_export.json (which has no request_latency)
    files = list(out_dir.glob("**/profile_export_aiperf.json"))
    if not files:
        raise FileNotFoundError(f"no profile_export_aiperf.json under {out_dir}")
    data = json.loads(files[0].read_text())
    # AIPerf schema: request_latency = {unit, avg, p1..p99, min, max, std, count, sum}
    lat = data.get("request_latency", {})
    ttft = data.get("time_to_first_token", {})   # absent for non-streaming; TTFT sweep needs streaming
    # aggregate throughput — the sweep's two throughput axes (output-token for out=100,
    # request for out=1). VERIFY keys on-box: recent AIPerf uses "output_token_throughput"
    # and "request_throughput" ({unit, avg}).
    tput = data.get("output_token_throughput", {})
    reqput = data.get("request_throughput", {})
    return {
        "p50_ms": float(lat.get("p50", 0.0)),
        "p95_ms": float(lat.get("p95", 0.0)),
        "ttft_ms": float(ttft.get("p50", 0.0)) if isinstance(ttft, dict) else 0.0,
        "throughput_tok_s": float(tput.get("avg", 0.0)) if isinstance(tput, dict) else 0.0,
        "req_throughput_req_s": float(reqput.get("avg", 0.0)) if isinstance(reqput, dict) else 0.0,
        "samples_ms": _request_latencies(out_dir),
    }


def _request_latencies(out_dir: Path) -> list[float]:
    """Best-effort per-request latencies (ms) for the full trace.

    The aggregated metrics file holds only percentiles; the raw per-request records
    live in profile_export.json (one entry per request). VERIFY on-box: the record
    field name and unit (AIPerf typically reports latency in ns). Returns [] if the
    raw file/schema isn't present so the aggregate (p50/p95) path still works.
    """
    raw = [f for f in out_dir.glob("**/profile_export.json")
           if f.name != "profile_export_aiperf.json"]
    if not raw:
        return []
    try:
        records = json.loads(raw[0].read_text())
    except (OSError, ValueError):
        return []
    if isinstance(records, dict):                       # VERIFY: some versions wrap in {"requests": [...]}
        records = records.get("requests") or records.get("experiments") or []
    out: list[float] = []
    for rec in records if isinstance(records, list) else []:
        val = rec.get("request_latency") if isinstance(rec, dict) else None
        if isinstance(val, (int, float)):
            out.append(round(val / 1e6, 3))             # VERIFY unit: ns -> ms
    return out


# vLLM-Omni generation endpoints (from the live server's OpenAPI):
#   image  -> POST /v1/images/generations  (JSON, returns data[0].b64_json)
#   video  -> POST /v1/videos/sync         (multipart/form-data, returns the MP4 directly)
GEN_PROMPT = "a robot arm picking up a red cube on a table"
GEN_FPS = 24   # report: 189 frames @ 24 FPS


def _size_for(op: OperatingPoint) -> str:
    """Resolution tag -> "WxH". VERIFY Cosmos3's supported sizes per tier (a 400 lists them)."""
    return {"1024px": "1024x1024", "256p": "256x256",
            "480p": "832x480", "720p": "1280x720"}.get(op.clip_resolution, "512x512")


def _encode_multipart(fields: dict) -> tuple[bytes, str]:
    """Minimal multipart/form-data encoder (stdlib only) for scalar form fields."""
    boundary = "----cosmosbench7f3a2b"
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{k}"\r\n\r\n{v}\r\n').encode()
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _generate_once(base_url: str, model: str, op: OperatingPoint) -> None:
    """One blocking generation, timed by the caller. Reads the full response so the
    elapsed time is the full generation (both endpoints return the finished media)."""
    if op.modality == "image":
        body = json.dumps({"model": model, "prompt": GEN_PROMPT, "n": 1,
                           "size": _size_for(op), "seed": 0}).encode()
        req = urllib.request.Request(f"{base_url}/v1/images/generations", data=body,
                                     headers={"Content-Type": "application/json"})
    else:
        # T2V / I2V -> synchronous video endpoint (multipart). VERIFY: true I2V (i2v-*) wants
        # an `image_reference` file part; we send text->video at the same size/frames, whose
        # denoise cost is ~identical, so the latency benchmark holds. 189 frames @ 24 FPS.
        fields = {"model": model, "prompt": GEN_PROMPT, "size": _size_for(op),
                  "num_frames": op.clip_frames, "fps": GEN_FPS, "seed": 0}
        body, ctype = _encode_multipart(fields)
        req = urllib.request.Request(f"{base_url}/v1/videos/sync", data=body,
                                     headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=3600) as r:
        r.read()


def time_generation_request(base_url: str, model: str, op: OperatingPoint,
                            *, repeats: int = 5, warmup: int = 1) -> dict:
    """Generator latency: wall-clock per clip over `repeats` (fixed prompt/seed)."""
    for _ in range(warmup):
        _generate_once(base_url, model, op)
    samples_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _generate_once(base_url, model, op)
        samples_ms.append((time.perf_counter() - t0) * 1e3)

    trace = [round(s, 3) for s in samples_ms]   # raw per-repeat order for the full trace
    samples_ms.sort()
    p95 = samples_ms[min(len(samples_ms) - 1, round(0.95 * (len(samples_ms) - 1)))]
    return {"p50_ms": statistics.median(samples_ms), "p95_ms": p95, "samples_ms": trace}


def measure_generation_throughput(base_url: str, model: str, op: OperatingPoint,
                                  *, batch: int, warmup: int = 1, waves: int = 2) -> float:
    """Generator throughput (clips/s) at a given batch size, for the batching sweep.

    Fires `batch` generation requests concurrently and times the whole wave; vLLM-Omni
    continuous-batches them (report §5.3.1). Returns clips/s = requests / wall.
    """
    import concurrent.futures as cf

    def _wave() -> float:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=batch) as ex:
            list(ex.map(lambda _i: _generate_once(base_url, model, op), range(batch)))
        return time.perf_counter() - t0

    for _ in range(warmup):
        _wave()
    per_wave_s = statistics.median(_wave() for _ in range(max(1, waves)))
    return batch / per_wave_s if per_wave_s > 0 else 0.0
