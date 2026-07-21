# syntax=docker/dockerfile:1
# LichtFeld Studio container-service image. Ported from the superseded
# SFMAPI/sfmapi_lfs repo's root Dockerfile (decision D3); the engine
# build (MrNeRF/LichtFeld-Studio, ref arg SFMAPI_LFS_REF, default
# master = the upstream default branch) is unchanged — only the
# plugin-package install and CMD moved to the merged 3dgs
# package.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG SFMAPI_LFS_REF=master
ARG UV_VERSION=0.8.15
ARG CMAKE_VERSION=4.0.3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SFMAPI_LFS_ROOT=/opt/LichtFeld-Studio \
    SFMAPI_PLUGIN_OUTPUT_ROOT=/sfmapi/output \
    SFMAPI_PLUGIN_WORK_ROOT=/sfmapi/work \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    DEBIAN_FRONTEND=noninteractive \
    VCPKG_ROOT=/opt/vcpkg

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git unzip zip tar pkg-config python3 python3-pip python3-venv python3-dev \
    build-essential gcc-14 g++-14 gfortran-14 ccache ninja-build nasm \
    autoconf autoconf-archive automake libtool \
    libxinerama-dev libxcursor-dev xorg-dev libglu1-mesa-dev \
    libwayland-dev libxkbcommon-dev libegl-dev libdecor-0-dev libibus-1.0-dev libdbus-1-dev libsystemd-dev libgtk-3-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 60 \
    && rm -rf /var/lib/apt/lists/*

RUN arch="$(uname -m)" \
    && curl -fsSL -o /tmp/cmake.sh "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${arch}.sh" \
    && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake.sh

RUN git clone https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics

RUN git clone --recursive https://github.com/MrNeRF/LichtFeld-Studio.git /opt/LichtFeld-Studio \
    && cd /opt/LichtFeld-Studio \
    && git checkout ${SFMAPI_LFS_REF} \
    && cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLFS_ENFORCE_LINUX_GUI_BACKENDS=OFF \
    && cmake --build build --config Release --target LichtFeld-Studio -j"$(nproc)"

RUN curl -LsSf https://astral.sh/uv/${UV_VERSION}/install.sh | sh
ENV PATH="/opt/sfmapi-venv/bin:/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/sfmapi-venv \
    && uv pip install --python /opt/sfmapi-venv/bin/python .

EXPOSE 8080
CMD ["sfmapi-lfs", "--host", "0.0.0.0", "--port", "8080"]
