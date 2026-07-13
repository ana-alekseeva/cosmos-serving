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

from bench.workload import OperatingPoint


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
        # VERIFY: default is 1 image/request; confirm the flag that sets image count if needed.
        cmd += ["--image-width-mean", str(w), "--image-height-mean", str(h)]
    elif op.modality == "video":
        w, h = _res_to_wh(op.clip_resolution)
        # VERIFY on-box: AIPerf's synthetic-VIDEO support + exact flags. `clip_frames` encodes
        # the 1-vs-2 FPS distinction (NVIDIA "Video 1/2 FPS"). If AIPerf can't synthesize video,
        # supply a fixed sample clip of `op.clip_frames` frames at (w,h) via --input-file, or pass
        # the frame count through to the request with an extra input.
        cmd += ["--image-width-mean", str(w), "--image-height-mean", str(h),
                "--extra-inputs", f"num_frames:{op.clip_frames}"]   # VERIFY payload key
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


def time_generation_request(base_url: str, model: str, op: OperatingPoint,
                            *, repeats: int = 5, warmup: int = 1) -> dict:
    """Generator latency: wall-clock per clip over `repeats` (fixed prompt/seed)."""
    payload = json.dumps({
        "model": model,
        "prompt": "a robot arm picking up a red cube on a table",   # fixed prompt/seed
        "seed": 0,
        "resolution": op.clip_resolution,                            # VERIFY payload schema
        "num_frames": op.clip_frames,
    }).encode()

    def _one() -> None:
        req = urllib.request.Request(f"{base_url}/v1/generate",       # VERIFY endpoint path
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            r.read()

    for _ in range(warmup):
        _one()
    samples_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _one()
        samples_ms.append((time.perf_counter() - t0) * 1e3)

    trace = [round(s, 3) for s in samples_ms]   # raw per-repeat order for the full trace
    samples_ms.sort()
    p95 = samples_ms[min(len(samples_ms) - 1, round(0.95 * (len(samples_ms) - 1)))]
    return {"p50_ms": statistics.median(samples_ms), "p95_ms": p95, "samples_ms": trace}


def measure_generation_throughput(base_url: str, model: str, op: OperatingPoint,
                                  *, batch: int, warmup: int = 1, waves: int = 2) -> float:
    """Generator throughput (clips/s) at a given batch size, for the batching sweep.

    Fires `batch` generation requests concurrently and times the whole wave; vLLM-Omni
    continuous-batches / seq-packs them (report §5.3.1). Returns clips/s = requests / wall.
    """
    import concurrent.futures as cf

    payload = json.dumps({
        "model": model,
        "prompt": "a robot arm picking up a red cube on a table",
        "seed": 0,
        "resolution": op.clip_resolution,                            # VERIFY payload schema
        "num_frames": op.clip_frames,
    }).encode()

    def _one(_i: int) -> None:
        req = urllib.request.Request(f"{base_url}/v1/generate",       # VERIFY endpoint path
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            r.read()

    def _wave() -> float:
        t0 = time.perf_counter()
        with cf.ThreadPoolExecutor(max_workers=batch) as ex:
            list(ex.map(_one, range(batch)))
        return time.perf_counter() - t0

    for _ in range(warmup):
        _wave()
    per_wave_s = statistics.median(_wave() for _ in range(max(1, waves)))
    return batch / per_wave_s if per_wave_s > 0 else 0.0
