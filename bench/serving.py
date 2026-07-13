"""Launch / stop a vLLM (Reasoner) or vLLM-Omni (Generator) server for a given
technique set — the real backend's server lifecycle.

IMPORTANT: written against the documented vLLM / vLLM-Omni CLI, but NOT yet run on a
GPU. Every mapping marked `# VERIFY` must be confirmed on-box against the installed
version and `recipes/cosmos3/Cosmos3-Nano.md`. The mock backend needs none of this.

Two very different launches:
  - Reasoner: ONE stock-vLLM config, reused for the whole concurrency sweep (§5.3.2 —
    the report inherits Qwen3-VL serving out of the box). Concurrency is varied by
    AIPerf per request, not by a server flag. `techniques` is always empty here.
  - Generator: one vLLM-Omni server per ablation variant; each enabled technique adds
    its flags, and CUDA graphs is a real toggle (baseline is --enforce-eager).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from optimize.registry import GENERATOR, REASONER, Technique

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"

# Pin the run so variants/points are comparable and low-noise (spec N1/N2).
DTYPE = "bfloat16"
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.90
SEED = 0

# Generator (vLLM-Omni) technique -> server args, added when the technique is ENABLED.
# VERIFY exact names on-box against recipes/cosmos3/Cosmos3-Nano.md.
_ENABLE_ARGS: dict[str, list[str]] = {
    "fp8":              ["--quantization", "fp8"],
    "cache-dit":        ["--cache-dit"],
    "vae-patch":        ["--vae-patch-parallel"],
    "cfg-parallel":     ["--cfg-parallel"],
    "context-parallel": ["--ulysses-degree", "2"],
    "hsdp":             ["--hsdp"],
    "cpu-offload":      ["--enable-layerwise-offload"],
}
_MULTI_GPU = {"cfg-parallel", "context-parallel"}  # need 2 GPUs


def build_command(model: str, tower: str, techniques: list[Technique],
                  port: int) -> tuple[list[str], dict, int]:
    """Build the vLLM launch command + env for a tower / technique set."""
    keys = {t.key for t in techniques}
    cmd = ["vllm", "serve", model, "--host", "0.0.0.0", "--port", str(port),
           "--dtype", DTYPE, "--max-model-len", str(MAX_MODEL_LEN),
           "--gpu-memory-utilization", str(GPU_MEM_UTIL), "--seed", str(SEED)]
    env: dict[str, str] = {}

    if tower == REASONER:
        # Stock vLLM, out of the box: paged KV-cache, continuous batching, fused
        # attention, prefix caching are all ON by default — that IS the config the
        # report uses. No ablation toggles here; the sweep varies concurrency via AIPerf.
        return cmd, env, 1

    # Generator (vLLM-Omni).
    cmd.append("--omni")
    if "cuda-graphs" not in keys:               # torch.compile + CUDA graphs: baseline is eager
        cmd.append("--enforce-eager")
    for k in sorted(keys):
        cmd += _ENABLE_ARGS.get(k, [])
    n_gpus = 2 if keys & _MULTI_GPU else 1
    if n_gpus > 1:
        cmd += ["--tensor-parallel-size", "2"]
    return cmd, env, n_gpus


@dataclass
class ServerHandle:
    proc: subprocess.Popen
    base_url: str
    log_path: str | None = None

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def start_server(model: str, tower: str, techniques: list[Technique],
                 port: int = 8000, ready_timeout: int = 1800) -> ServerHandle:
    if shutil.which("vllm") is None:
        raise RuntimeError(
            "`vllm` not found — the real backend runs on a GPU host. "
            "Install it with deploy/setup_gpu.sh, or use --backend mock locally."
        )
    cmd, env, _ = build_command(model, tower, techniques, port)
    log_path = tempfile.NamedTemporaryFile(prefix="vllm-serve-", suffix=".log", delete=False).name
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, env={**os.environ, **env},
                            stdout=log, stderr=subprocess.STDOUT)  # captured, tailed on failure
    handle = ServerHandle(proc, f"http://127.0.0.1:{port}", log_path)
    # vLLM startup is silent for 1-2 min (load + compile + memory profiling); announce it
    # so a slow-but-healthy launch doesn't read as a hang. `tail -f` the log for detail.
    print(f"    launching vLLM — waiting for /health (up to {ready_timeout // 60} min); "
          f"log: {log_path}", flush=True)
    t0 = time.time()
    try:
        _wait_healthy(handle.base_url, proc, ready_timeout)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{exc}\n  cmd: {' '.join(cmd)}\n  --- vllm log tail ({log_path}) ---\n{_tail(log_path)}"
        ) from None
    print(f"    ✓ vLLM healthy in {time.time() - t0:.0f}s", flush=True)
    return handle


def _tail(path: str, n: int = 20) -> str:
    try:
        return "".join(open(path).readlines()[-n:]) or "(empty log)"
    except OSError:
        return "(log unavailable)"


def _wait_healthy(base_url: str, proc: subprocess.Popen, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (exit code {proc.returncode})")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(3)
    raise TimeoutError(f"server not healthy after {timeout}s: {base_url}")
