# syntax=docker/dockerfile:1
#
# SigLIP image embedder — standalone, platform-agnostic.
#
# A FastAPI HTTP server wrapping a SigLIP / SigLIP 2 image+text encoder.
# Model weights are baked into the image (~1 GB on top of the PyTorch CUDA
# base) so the container starts cleanly on any host with an NVIDIA GPU and
# does not need volumes, init containers, or external storage.
#
# Build:
#   docker build -t ghcr.io/<you>/siglip-embedder:latest \
#     --build-arg MODEL_ID=google/siglip2-so400m-patch14-384 \
#     .

FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ARG MODEL_ID=google/siglip2-so400m-patch14-384

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf \
    TRANSFORMERS_CACHE=/opt/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    SIGLIP_MODEL=${MODEL_ID}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install hf_transfer \
    && pip install -r requirements.txt

# Bake the model into the image so cold start does not pull ~1 GB from the
# Hugging Face hub. hf_transfer parallelizes the download. For gated repos,
# pass the token via build secret and uncomment the mount line below.
# RUN --mount=type=secret,id=HF_TOKEN HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='${MODEL_ID}', local_dir='/opt/hf/models/${MODEL_ID}', \
local_dir_use_symlinks=False, allow_patterns=['*.json','*.txt','*.safetensors','*.model','*.bin'])"

ENV SIGLIP_MODEL_PATH=/opt/hf/models/${MODEL_ID}

COPY server.py .
COPY --chmod=755 run.sh /run.sh

EXPOSE 8000

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/run.sh"]
