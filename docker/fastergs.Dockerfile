# syntax=docker/dockerfile:1
# Faster-GS container-service image. Ported from the superseded
# SFMAPI/sfmapi_fastergs repo's root Dockerfile (decision D3); the
# engine checkouts (nerficg-project/nerficg + faster-gaussian-splatting,
# ref args SFMAPI_NERFICG_REF / SFMAPI_FASTERGS_REF) are unchanged —
# only the plugin-package install and CMD moved to the merged
# sceneapi-3dgs package.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG SFMAPI_FASTERGS_REF=main
ARG SFMAPI_NERFICG_REF=main
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG CUDA_ARCHITECTURES=120
ARG UV_VERSION=0.8.15

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SFMAPI_FASTERGS_ROOT=/opt/fastergs \
    SFMAPI_FASTERGS_FRAMEWORK_ROOT=/opt/nerficg \
    SFMAPI_PLUGIN_OUTPUT_ROOT=/sfmapi/output \
    SFMAPI_PLUGIN_WORK_ROOT=/sfmapi/work \
    TORCH_HOME=/opt/sfmapi/torch-cache \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential ninja-build cmake \
    libgl1 libglib2.0-0 libxrender1 libxext6 libfreetype6 libfontconfig1 \
    libx11-6 libxcursor1 libxinerama1 libxi6 libxrandr2 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/root/.local/bin:/opt/conda/bin:${PATH}"

RUN git clone --recursive https://github.com/nerficg-project/nerficg.git /opt/nerficg \
    && cd /opt/nerficg \
    && git checkout ${SFMAPI_NERFICG_REF} \
    && git submodule update --init --recursive

RUN git clone --recursive https://github.com/nerficg-project/faster-gaussian-splatting.git /opt/fastergs \
    && cd /opt/fastergs \
    && git checkout ${SFMAPI_FASTERGS_REF} \
    # FasterGSCudaBackend headers include <torch/extension.h> (pulls pybind11);
    # build against the lighter <torch/types.h> instead. Idempotent: a no-op
    # if upstream already uses torch/types.h.
    && find FasterGSCudaBackend -name '*.h' -exec sed -i 's|#include <torch/extension.h>|#include <torch/types.h>|g' {} + \
    && mkdir -p /opt/nerficg/src/Methods \
    && ln -s /opt/fastergs /opt/nerficg/src/Methods/FasterGS

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "numpy<2.0" PyYAML munch natsort tqdm opencv-python kornia torchmetrics lpips einops matplotlib timm plotly imgui-bundle PyOpenGL "cuda-python<13" numpy-quaternion pysdl3 pillow "jax<0.5" pyproj scikit-learn "pycolmap>=3.11" plyfile \
    && uv pip install --system --no-build-isolation /opt/nerficg/src/CudaUtils/MortonEncoding \
    && uv pip install --system --no-build-isolation /opt/fastergs/FasterGSCudaBackend \
    && uv pip install --system --no-cache --no-build-isolation "simple_knn @ git+https://github.com/camenduru/simple-knn.git@60f461f4a56b7967e5d8045bf92f8c33f36976d0" \
    && uv pip install --system --no-cache --no-build-isolation "fused_ssim @ git+https://github.com/rahul-goel/fused-ssim.git@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"

RUN mkdir -p "${TORCH_HOME}" \
    && python - <<'PY'
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

for net in ("alex", "vgg"):
    LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True)
PY

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

EXPOSE 8080
CMD ["sfmapi-fastergs", "--host", "0.0.0.0", "--port", "8080"]
