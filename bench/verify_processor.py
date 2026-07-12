"""On-box check: does the Cosmos3 processor accept our multimodal input format?

Run on the GPU host — fast, loads only the PROCESSOR, not the 16B model:
    uv run python -m bench.verify_processor

Verdict tells you whether bench/eager.py::_build_inputs is right as-is (Pattern A),
or whether the processor wants the two-step Qwen-VL form (Pattern B).
"""
from __future__ import annotations


def main(model_id: str = "nvidia/Cosmos3-Nano") -> None:
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    tmpl = getattr(proc, "chat_template", None) or ""
    print("=== chat_template (first 600 chars — shows how it reads image/video content) ===")
    print(tmpl[:600] or "(no chat_template attribute)")

    gray = Image.new("RGB", (256, 256), (127, 127, 127))
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": gray},
        {"type": "text", "text": "What action should the robot take?"},
    ]}]

    # Pattern A — what bench/eager.py currently does: one call does text + vision.
    print("\n=== Pattern A: apply_chat_template(tokenize=True, return_dict=True) ===")
    try:
        out = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                       tokenize=True, return_dict=True, return_tensors="pt")
        vision = [k for k in out if k not in ("input_ids", "attention_mask")]
        print("  accepted. keys:", list(out.keys()))
        print("  VISION TENSORS:", vision or "NONE  <-- image ignored; use Pattern B")
        if vision:
            print("  ==> Pattern A works; _build_inputs is correct as-is. ✅")
    except Exception as e:
        print("  REJECTED:", type(e).__name__, "-", str(e)[:300])
        print("  (read the error — it usually names the expected content format)")

    # Pattern B — Qwen-VL classic: template -> text, then processor(text, images=...).
    print("\n=== Pattern B: apply_chat_template(tokenize=False) then processor(text, images=[...]) ===")
    try:
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        out = proc(text=[text], images=[gray], return_tensors="pt")
        print("  accepted. keys:", list(out.keys()))
        print("  ==> if Pattern A showed NO vision tensors, switch _build_inputs to this two-step form.")
    except Exception as e:
        print("  REJECTED:", type(e).__name__, "-", str(e)[:300])


if __name__ == "__main__":
    main()
