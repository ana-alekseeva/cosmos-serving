# Cosmos3-Nano-Policy-DROID specification

# 1. Scope

**Model:** `Cosmos3-Nano-Policy-DROID`

**Only evaluated inference task:**

```text
DROID camera observations
+ language instruction
+ proprioceptive state
→ 32 × 8 robot-action chunk
```

The model is evaluated by executing its generated actions in RoboLab.

Excluded:

* Standalone image-and-text-to-text generation.
* Text-to-video.
* Joint video generation.
* Techniques not described in the Cosmos 3 paper.
* Throughput batching.

The paper identifies native PyTorch as the modifiable reference backend and vLLM/vLLM-Omni as optimized serving backends. It also notes that batching does not benefit single-request, latency-bound robotics workloads. 

---

# 2. Required outputs

The project must produce:

1. Per-request latency logs for every optimization configuration.
2. Reasoner-stage and Generator-stage waterfall histograms.
3. Baseline-versus-optimized end-to-end latency breakdown.
4. RoboLab quality comparison for baseline and optimized models.
5. Production results using vLLM and vLLM-Omni.

---

# 3. Waterfall configurations

## Reasoner conditioning waterfall

The Reasoner is measured only as part of action-policy conditioning; it does not generate standalone text.

| ID   | Configuration                                      |
| ---- | -------------------------------------------------- |
| `R0` | Eager BF16 with math attention                     |
| `R1` | Add Flash/fused attention                          |
| `R2` | Add `torch.compile`                                |
| `R3` | Add CUDA graph replay                              |
| `R4` | Cache Reasoner conditioning across diffusion steps |

Reasoner caching must be enabled only after verifying that its output is invariant throughout one action-denoising trajectory. The cache must be invalidated for every new observation.

## Generator waterfall

| ID   | Configuration                  |
| ---- | ------------------------------ |
| `G0` | Eager BF16 with math attention |
| `G1` | Add Flash/fused attention      |
| `G2` | Add `torch.compile`            |
| `G3` | Add CUDA graph replay          |
| `G4` | Add Cache-DiT                  |
| `G5` | Add dynamic FP8 quantization   |

Cache-DiT and FP8 are quality-gated. They are included in the final configuration only when RoboLab performance remains acceptable.

## End-to-end cumulative waterfall

The main waterfall should use the combined action pipeline:

```text
Baseline eager
→ Flash Attention
→ torch.compile
→ CUDA graphs
→ Reasoner conditioning cache
→ Cache-DiT
→ FP8
→ Final optimized latency
```

Also produce a stage breakdown for baseline and final:

```text
Preprocessing
Reasoner conditioning
Generator preparation
Action denoising
Action post-processing
Transport
```

## Separate multi-GPU experiment

Do not mix different GPU counts in the primary waterfall.

Run a separate multi-GPU job for:

* CFG parallelism, only if the action pipeline uses CFG.
* Ulysses context parallelism.

Compare against the best single-GPU configuration.

Do not include CPU offload, HSDP, or VAE patch parallelism in the main latency waterfall. They are memory-oriented or inapplicable to action output.

---

# 4. Simplified Nebius job plan

## Job 1 — PyTorch ablation matrix

**Resources:** one target inference GPU.

Runs all single-GPU configurations sequentially as subprocesses:

```text
R0–R4
G0–G5
combined end-to-end configurations
```

Each subprocess:

1. Loads the model from local storage.
2. Runs warm-up requests.
3. Runs the fixed latency dataset.
4. Writes logs and summary files.
5. Exits and releases its CUDA context.

Recommended command structure:

```bash
python run_matrix.py \
  --config experiment.yaml \
  --checkpoint-dir /local/model \
  --input-manifest /local/replay/manifest.json \
  --output-dir /results
```

Internally:

```python
for configuration in configurations:
    subprocess.run(
        [
            "python",
            "run_configuration.py",
            "--configuration",
            configuration,
        ],
        check=True,
    )
```

Use a separate compilation-cache directory for every configuration:

```text
/tmp/torchinductor/R0
/tmp/torchinductor/R1
...
```

## Job 2 — Production-engine validation

This may be two jobs if vLLM and vLLM-Omni require incompatible environments:

* vLLM Reasoner validation.
* vLLM-Omni Generator/full-policy validation.

Run only:

* Production baseline.
* Final optimized production configuration.

Do not repeat the entire ablation matrix in production engines.

## Job 3 — RoboLab subset evaluation

Run:

* Baseline.
* Final optimized PyTorch.
* Final production configuration.

Prefer a RoboLab simulator job connected to a separate inference endpoint so rendering and physics do not compete with inference on the same GPU.

## Job 4 — Full RoboLab evaluation

Run only after the subset passes:

* Baseline.
* Final optimized production configuration.

## Job 5 — Aggregation

A small CPU job merges logs and generates:

* CSV or Parquet summaries.
* Waterfall data.
* Confidence intervals.
* Figures.
* Quality-comparison tables.

---

# 5. Latency dataset

## Offline replay set

Use a fixed set of approximately **256 requests captured from RoboLab episodes**.

Each request contains:

* Camera observations.
* Instruction.
* Proprioceptive state.
* Task and episode identifiers.
* Control timestep.
* Fixed inference seed.

Use this replay set for all waterfall measurements. This is substantially faster and less noisy than starting RoboLab for every configuration.

Synthetic tensors may be used only for attention microbenchmarks. The primary waterfall should use real RoboLab requests because preprocessing, masks, memory layouts, and prompt lengths affect latency.

## RoboLab quality subset

Use a stratified subset of approximately **18 tasks**:

```text
3 capability groups
× 3 difficulty levels
× 2 tasks
```

Run 10 episodes per task.

The subset is used to reject optimizations that damage policy performance. The complete RoboLab benchmark is needed only for baseline and final configurations.

---

# 6. Latency measurements

Every request should record:

| Field                  | Meaning                                       |
| ---------------------- | --------------------------------------------- |
| `preprocess_ms`        | Images, prompt, state, tensor preparation     |
| `h2d_ms`               | Host-to-GPU transfer                          |
| `reasoner_ms`          | Reasoner/conditioning computation             |
| `generator_prepare_ms` | Latent and schedule preparation               |
| `denoising_ms`         | Complete action diffusion loop                |
| `denoising_step_ms`    | Array of individual step timings              |
| `postprocess_ms`       | Action decode, denormalization, clipping      |
| `d2h_ms`               | GPU-to-host transfer                          |
| `server_ms`            | Server-side complete inference                |
| `transport_ms`         | Client/server communication                   |
| `first_action_ms`      | Observation ready to first action submission  |
| `total_chunk_ms`       | Observation ready to complete 32-action chunk |
| `peak_memory_mb`       | Peak allocated GPU memory                     |

Use:

* CUDA events for GPU stages.
* Monotonic CPU timers for end-to-end measurements.
* Batch size 1.
* Approximately 20–30 warm-ups.
* At least 200 measured requests per configuration.
* p50, p90, and p99 summaries.

Do not include:

* GPU provisioning.
* Container startup.
* Checkpoint download.
* Initial compilation.
* Dataset staging.

Record these separately as operational overhead.

---

# 7. Minimal log format

One JSONL row per request:

```json
{
  "run_id": "cosmos-droid-001",
  "configuration": "G3",
  "engine": "pytorch",
  "task": "RoboLabTask",
  "episode_id": 4,
  "request_id": 37,
  "latency_ms": {
    "preprocess": 7.1,
    "h2d": 1.2,
    "reasoner": 45.8,
    "generator_prepare": 2.4,
    "denoising": 116.3,
    "postprocess": 1.8,
    "d2h": 0.4,
    "server_total": 175.0,
    "first_action": 184.2,
    "chunk_total": 184.2
  },
  "denoising_step_ms": [5.9, 5.7, 5.8],
  "peak_memory_mb": 43120,
  "output_checksum": "...",
  "quality_gate": "passed"
}
```

Each subprocess also writes:

```text
summary.json
environment.json
system-info.json
status.json
```

---

# 8. Avoiding bias in a single long job

Running all configurations in one provisioned job is efficient, but account for GPU drift:

* Run the baseline at both the start and end.
* Randomize the order of intermediate configurations, or repeat the full sequence twice.
* Wait briefly between subprocesses.
* Record GPU temperature and clocks where available.
* Keep the same power mode and GPU type.
* Ensure no other process uses the GPU.
* Stage all inputs and model files locally before timing.

A useful sequence is:

```text
Baseline
→ configuration matrix in randomized order
→ Baseline repeated
```

Reject the run when the two baseline measurements differ substantially.

---

# 9. Main compatibility concerns

## Flash Attention with plain PyTorch

Use PyTorch SDPA and force the backend explicitly:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch.nn.functional as F

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
    )
```

Requirements:

* Fail instead of silently falling back.
* Log unsupported attention layers.
* Compare against forced math SDPA.
* Keep tensor layouts identical.
* Include transpose and contiguous-copy costs.
* Check custom masks, head dimensions, and tensor strides.

## `torch.compile` and CUDA graphs

Do not double-count CUDA graphs.

Measure:

```text
Eager
Compile without CUDA graphs
Compile with CUDA graphs
```

If `reduce-overhead` automatically enables graph replay, it must be treated as the combined configuration.

## Static shapes

CUDA graph configurations require fixed or bucketed:

* Image shapes.
* Prompt lengths.
* Batch size.
* Action chunk length.
* Diffusion-step count.
* Attention-mask shapes.

## Cache-DiT compatibility

Cache-DiT may introduce dynamic control flow that conflicts with CUDA graph capture. Test:

```text
CUDA graphs only
Cache-DiT only
CUDA graphs + Cache-DiT
```

Include the combination only when it works correctly and gives an additional benefit.

## FP8

FP8 requires supported hardware and kernels. It must pass:

* Offline action-difference checks.
* RoboLab subset evaluation.
* Full RoboLab validation before final acceptance.

---

# 10. Final acceptance

The optimized model is accepted when it has:

* Lower p50 and p99 action-chunk latency.
* Lower observation-to-first-action latency.
* Valid `32 × 8` action outputs.
* No meaningful RoboLab success regression.
* Reproducible logs from the fixed replay set.
* Successful production serving through vLLM/vLLM-Omni.

The most efficient infrastructure design is therefore **one provisioned PyTorch GPU job containing isolated subprocesses for the complete single-GPU waterfall**, followed by separate, much smaller jobs for production validation and RoboLab evaluation.
