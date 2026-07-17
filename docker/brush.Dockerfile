# syntax=docker/dockerfile:1
# Brush container-service image. Ported from the superseded
# SFMAPI/sfmapi_brush repo's root Dockerfile (decision D3); the engine
# build (ArthurBrussee/brush) is unchanged — only the plugin-package
# install and CMD moved to the merged sfmapi-radiance package.
FROM rust:1-bookworm

ARG SFMAPI_BRUSH_REF=main
ARG UV_VERSION=0.8.15

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SFMAPI_BRUSH_ROOT=/opt/brush \
    SFMAPI_PLUGIN_OUTPUT_ROOT=/sfmapi/output \
    SFMAPI_PLUGIN_WORK_ROOT=/sfmapi/work \
    WGPU_BACKEND=vulkan \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git python3 python3-pip python3-venv \
    build-essential pkg-config cmake clang libclang-dev libssl-dev \
    libvulkan1 vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/opt/sfmapi-venv/bin:/root/.local/bin:${PATH}"

RUN git clone --recursive https://github.com/ArthurBrussee/brush.git /opt/brush \
    && cd /opt/brush \
    && git checkout ${SFMAPI_BRUSH_REF} \
    && cargo build --release -p brush-app --bin brush

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/sfmapi-venv \
    && uv pip install --python /opt/sfmapi-venv/bin/python .

EXPOSE 8080
CMD ["sfmapi-brush", "--host", "0.0.0.0", "--port", "8080"]
