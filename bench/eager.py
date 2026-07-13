"""Eager reference-path backend — HuggingFace Transformers with per-technique toggles.

This is the ONLY engine where the fundamentals (KV cache, etc.) are real on/off
switches; vLLM bakes them in. It drives Cosmos3-Nano via `model.generate()` so
`kv-cache` (use_cache), `cuda-graphs` (torch.compile), `flash-attn`
(attn_implementation) and `fp8` (quantization) each toggle. Times each config with
CUDA events (the hw2 `time_generation` pattern).

NOT yet run on a GPU. Confirm each `# VERIFY` on-box — the model class, processor,
multimodal input building, and fp8 config — exactly like the vLLM backend needed.
Single-request OPs only (A/B/D/E); concurrency (C/F) is a batching concept, not eager's.
"""
from __future__ import annotations

import statistics
from contextlib import nullcontext

from bench.drivers import Measurement
from bench.workload import OperatingPoint
from optimize.registry import REASONER, Technique

DEFAULT_MODEL = "nvidia/Cosmos3-Nano"


def _res_to_px(res: str) -> tuple[int, int]:
    """Map an OP resolution tag to (width, height) for the synthetic frame. VERIFY dims."""
    return {"256p": (256, 256), "512px": (512, 512),
            "480p": (854, 480), "720p": (1280, 720)}.get(res, (256, 256))


class EagerEngine:
    backend = "eager"

    def __init__(self, enabled: list[Technique], *, tower: str = REASONER,
                 model: str | None = None, port: int | None = None):
        self.keys = {t.key for t in enabled}
        self.model_id = model or DEFAULT_MODEL
        self._model = None
        self._proc = None

    # -- lifecycle ---------------------------------------------------------------
    def _load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor  # VERIFY class name

        # baseline attention = "sdpa" (reference); flash-attn switches it on
        attn = "flash_attention_2" if "flash-attn" in self.keys else "sdpa"
        kwargs = dict(dtype=torch.bfloat16, attn_implementation=attn, device_map="cuda")
        if "fp8" in self.keys:
            # VERIFY: HF fp8 path, e.g. quantization_config=FineGrainedFP8Config()
            from transformers import FineGrainedFP8Config  # VERIFY import
            kwargs["quantization_config"] = FineGrainedFP8Config()

        model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs).eval()
        if "cuda-graphs" in self.keys:
            model = torch.compile(model, mode="reduce-overhead")   # torch.compile + CUDA graphs
        self._model = model
        self._proc = AutoProcessor.from_pretrained(self.model_id)

    def prepare(self) -> None:
        """Load the model once up front so a load failure dooms the variant fast."""
        if self._model is None:
            self._load()

    def close(self):
        import gc
        self._model = None
        self._proc = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # -- inputs ------------------------------------------------------------------
    def _build_inputs(self, op: OperatingPoint):
        # Synthetic input of the right SHAPE — latency is shape-driven, not content-driven,
        # so a solid-gray frame encodes to the same #vision-tokens as a real image/clip
        # (no media file needed). VERIFY on-box: the Cosmos3 processor's chat-template
        # content format (the "image"/"video" keys, PIL vs path/decoded array).
        if op.modality == "text":
            content = "Describe the scene." + " context" * max(0, op.input_tokens // 2)
        else:
            from PIL import Image
            gray = Image.new("RGB", _res_to_px(op.clip_resolution), (127, 127, 127))
            if op.modality == "image":
                media = [{"type": "image", "image": gray}]
            else:  # video: clip_frames identical synthetic frames
                media = [{"type": "video", "video": [gray] * max(1, op.clip_frames)}]
            content = media + [{"type": "text", "text": "What action should the robot take?"}]
        messages = [{"role": "user", "content": content}]
        return self._proc.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")

    # -- measurement -------------------------------------------------------------
    def measure(self, op: OperatingPoint, repeats: int = 3, warmup: int = 1) -> Measurement:
        import torch

        if op.concurrency > 1:
            raise RuntimeError(
                f"eager backend is single-request; OP {op.name} has concurrency "
                f"{op.concurrency} (that's a batching concept — measure it on vLLM)."
            )
        if self._model is None:
            self._load()

        inputs = self._build_inputs(op)
        gen_kwargs = dict(max_new_tokens=op.output_tokens, do_sample=False,
                          use_cache="kv-cache" in self.keys)   # <-- the KV-cache toggle

        def _one():
            with torch.inference_mode():
                self._model.generate(**inputs, **gen_kwargs)

        for _ in range(warmup):        # pays compile/autotune + CUDA init before timing
            _one()
        torch.cuda.synchronize()

        samples_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _one()
            end.record()
            torch.cuda.synchronize()
            samples_ms.append(start.elapsed_time(end))

        trace = tuple(round(s, 3) for s in samples_ms)   # preserve raw per-repeat order for the full trace
        samples_ms.sort()
        p95 = samples_ms[min(len(samples_ms) - 1, round(0.95 * (len(samples_ms) - 1)))]
        return Measurement(op.name, round(statistics.median(samples_ms), 3), round(p95, 3), trace)
