# Cosmos3-Nano-Policy-DROID benchmark image — NATIVE-PyTorch (cu130) runtime.
#
# cosmos-framework declares `vllm` and `cu130` as CONFLICTING uv groups, so ONE venv can't have
# both. This image is the native (cu130) runtime that runs the P0-P3 ladder (§5.3.1); the vLLM
# E-ladder + jobs 2/2b need a SEPARATE `--group vllm` image. All indexes are public (pytorch.org,
# nvidia-cosmos.github.io, whl.natten.org) so no build secret is needed. Build for linux/amd64.
#
# Bakes cuDNN >= 9.22 at build time (the cu130 group pins 9.15.1, which is too old) so the P
# cuDNN fused-attention baseline never hits "Could not find a compatible Attention backend". Venvs:
#   /opt/cosmos-serving/.venv        harness — mock / DROID capture (tfds)
#   /opt/cosmos-framework/.venv      model  — torch cu130 + cosmos_framework + this harness (editable)
#
# NOT baked (needs secrets / a GPU / is huge) — do these at RUNTIME on the GPU box:
#   * model weights  -> hf download <MODEL> --local-dir /local/model      (needs HF_TOKEN)
#   * replay set     -> policy.capture --n 50 --out /local/replay
#   * the matrix     -> run_matrix.py ...  (see the runtime block at the bottom)
#
# Build (context = repo root):
#   docker build -f deploy/Dockerfile -t <registry>/cosmos-droid-bench:latest .
#   docker push  <registry>/cosmos-droid-bench:latest
# Then point your npa workbench / job image at that tag.
#
# Every `# VERIFY` is environment-specific — confirm against your Nebius project before trusting it.

# CUDA 13 devel (nvcc present for any CUDA-extension builds during `uv sync`). VERIFY the tag
# exists / matches the torch cu13x build cosmos-framework pins.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

# cosmos-framework revision to pin (VERIFY: private repo? then add a build secret for the clone).
ARG COSMOS_FRAMEWORK_REPO=https://github.com/NVIDIA/cosmos-framework
ARG COSMOS_FRAMEWORK_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/root/.local/bin:${PATH}" \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # activate the model venv by default (that's where --backend pytorch runs)
    VENV_HARNESS=/opt/cosmos-serving/.venv \
    VENV_MODEL=/opt/cosmos-framework/.venv

# System deps + uv (Python/venv manager; it also provides the 3.13 interpreter).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv --version

WORKDIR /opt

# The harness (this repo). .dockerignore keeps .venv/.git/results/.nebius-physical-ai out.
COPY . /opt/cosmos-serving

# cosmos-framework (the model stack) — sibling dir, same layout deploy/setup.sh uses.
RUN git clone --depth 1 --branch "${COSMOS_FRAMEWORK_REF}" "${COSMOS_FRAMEWORK_REPO}" /opt/cosmos-framework

# 1) Harness venv: mock / capture (tfds+tf-cpu via the `capture` extra) / vLLM tooling.
RUN cd /opt/cosmos-serving && uv sync --extra capture

# 2) Model venv for the NATIVE-PyTorch path (§5.3.1): the `cu130` group (torch 2.10.0+cu130 +
#    flash-attn/natten from the cosmos index + cosmos_framework), plus `policy-server`, plus this
#    harness editable so `import policy` works. NOTE: cosmos-framework declares `vllm` and `cu130`
#    as CONFLICTING uv groups ([tool.uv].conflicts) — one venv can't have both, so this image is the
#    native (cu130) runtime; the vLLM E-ladder / jobs 2/2b need a separate `--group vllm` env.
#    The `cu130` group pins nvidia-cudnn-cu13==9.15.1 (< 9.22), so we still bump cuDNN to enable the
#    fused-attention backend (torch loads cuDNN from the wheel; the bump raises cudnn.version() at
#    runtime — torch.cuda.is_available() is False during build, so it only flips True on the GPU box).
RUN cd /opt/cosmos-framework \
    && uv sync --group cu130 --group policy-server \
    && uv pip install -e /opt/cosmos-serving \
    && cuver="$(uv run python -c 'import torch; print(torch.backends.cudnn.version() or 0)')" \
    && cumaj="$(uv run python -c 'import torch; print((torch.version.cuda or "13").split(".")[0])')" \
    && echo "cuDNN in image: ${cuver} (cu${cumaj} torch build)" \
    && if [ "${cuver:-0}" -lt 92200 ]; then \
         echo "bumping nvidia-cudnn-cu${cumaj}>=9.22 for the cuDNN fused-attention backend"; \
         uv pip install -U "nvidia-cudnn-cu${cumaj}>=9.22"; \
         uv run python -c 'import torch; print("cuDNN baked:", torch.backends.cudnn.version())'; \
       fi

# Fast-scratch mount point (model weights + replay set land here at runtime).
RUN mkdir -p /local/model /local/replay

WORKDIR /opt/cosmos-serving

# ---------------------------------------------------------------------------------------------
# RUNTIME (on the GPU box; provide HF_TOKEN, and AWS_* if uploading). The image is just the env —
# npa / your job runs these. Activate the HARNESS venv for capture, the MODEL venv for the matrix:
#
#   export HF_TOKEN=...   # accept the Cosmos3-Nano-Policy-DROID license on HF first
#   hf auth login --token "$HF_TOKEN" --add-to-git-credential
#   MODEL=$(sed -nE 's/^[[:space:]]+model:[[:space:]]*//p' config/experiment.yaml | head -1)
#   hf download "$MODEL" --local-dir /local/model
#
#   # capture the replay set in the HARNESS venv (has tfds)
#   source /opt/cosmos-serving/.venv/bin/activate && cd /opt/cosmos-serving
#   python -m policy.capture --n 50 --out /local/replay
#
#   # run the matrix in the MODEL venv (torch + cosmos_framework + policy)
#   source /opt/cosmos-framework/.venv/bin/activate
#   python run_matrix.py --backend pytorch --checkpoint-dir /local/model \
#     --input-manifest /local/replay/manifest.json --replay-size 50 --warmups 5 --output-dir results
#   python aggregate.py --out-dir results
# ---------------------------------------------------------------------------------------------
CMD ["/bin/bash"]
