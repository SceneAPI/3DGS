# syntax=docker/dockerfile:1
# spirulae-splat container-service image. Ported from the superseded
# SFMAPI/sfmapi_spirulae repo's root Dockerfile (decision D3); the
# engine checkout + patches (harry7557558/spirulae-splat, ref arg
# SFMAPI_SPIRULAE_REF, default master = the upstream default branch)
# are unchanged — only the plugin-package install and CMD moved to the
# merged sceneapi-3dgs package.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ARG SFMAPI_SPIRULAE_REF=master
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG UV_VERSION=0.8.15

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SFMAPI_SPIRULAE_ROOT=/opt/spirulae-splat \
    SFMAPI_PLUGIN_OUTPUT_ROOT=/sfmapi/output \
    SFMAPI_PLUGIN_WORK_ROOT=/sfmapi/work \
    TORCH_HOME=/opt/sfmapi/torch-cache \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
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

RUN git clone --recursive https://github.com/harry7557558/spirulae-splat.git /opt/spirulae-splat \
    && cd /opt/spirulae-splat \
    && git checkout ${SFMAPI_SPIRULAE_REF}

RUN python -c 'from pathlib import Path; p=Path("/opt/spirulae-splat/setup.py"); text=p.read_text(); old="    else:\n        raise RuntimeError(\"CUDA is required for this extension.\")\n"; new="    else:\n        env_arches = os.getenv(\"TORCH_CUDA_ARCH_LIST\", \"\")\n        if not env_arches:\n            raise RuntimeError(\"CUDA is required for this extension.\")\n        cuda_arch_list.extend(arch.strip().replace(\".\", \"\") for arch in env_arches.replace(\" \", \";\").split(\";\") if arch.strip())\n"; p.write_text(text.replace(old, new))'

RUN python <<'PY'
from pathlib import Path

p = Path("/opt/spirulae-splat/spirulae_splat/modules/model.py")
text = p.read_text()
lines = text.splitlines()
for idx, line in enumerate(lines):
    if f"s.replace('{chr(92)}033[m', '')" in line:
        escape = chr(92) + "033"
        lines[idx] = f'            clean = s.replace("{escape}[m", "")'
        lines.insert(
            idx + 1,
            f'            return f"{escape}[1;41m{{clean}}{escape}[m" if threshold >= 1.0 else s',
        )
        text = "\n".join(lines) + "\n"
        break
else:
    raise RuntimeError("expected Spirulae redbkg snippet was not found")
old = """            f"{redbkg(f'{self.overfit_count}/{self.config.early_stop_patience}',
                self.overfit_count/(0.8*self.config.early_stop_patience))}"""
new = """            redbkg(
                str(self.overfit_count) + '/' + str(self.config.early_stop_patience),
                self.overfit_count/(0.8*self.config.early_stop_patience),
            )"""
if old not in text:
    raise RuntimeError("expected Spirulae validation status snippet was not found")
patched = text.replace(old, new)
patched = patched.replace("""            )"
        ]""", """            )
        ]""")
patched_lines = patched.splitlines()
for idx, line in enumerate(patched_lines[:-1]):
    if "overfit_score" in line and patched_lines[idx + 1].strip() == "redbkg(":
        if not line.rstrip().endswith(","):
            patched_lines[idx] = f"{line},"
        patched = "\n".join(patched_lines) + "\n"
        break
else:
    raise RuntimeError("expected Spirulae validation overfit line was not found")
p.write_text(patched)

opaque_path = Path("/opt/spirulae-splat/spirulae_splat/strategy/opaque.py")
opaque_text = opaque_path.read_text()
opaque_old = 'get_param_attr(params, "means")'
if opaque_old not in opaque_text:
    raise RuntimeError("expected Spirulae opaque strategy snippet was not found")
opaque_path.write_text(opaque_text.replace(opaque_old, "get_param_attr(params, 'means')"))

benchmark_path = Path("/opt/spirulae-splat/spirulae_splat/ss_benchmark.py")
benchmark_text = benchmark_path.read_text()
benchmark_old = 'numalign="left"'
if benchmark_old not in benchmark_text:
    raise RuntimeError("expected Spirulae benchmark tabulate snippet was not found")
benchmark_path.write_text(benchmark_text.replace(benchmark_old, "numalign='left'"))

export_path = Path("/opt/spirulae-splat/scripts/export_ply_3dgs.py")
export_text = export_path.read_text()
sky_block = """    print()

    sky_path = os.path.join(os.path.dirname(output_path), "background.png")

    print("Exporting sky...")
    export_equirectangular(model, sky_path)
    print("Sky map saved to", sky_path)
"""
if sky_block not in export_text:
    raise RuntimeError("expected Spirulae sky export block was not found")
skip_sky_block = """    print()
    print("Skipping optional sky export.")
"""
export_path.write_text(export_text.replace(sky_block, skip_sky_block))
PY

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system "numpy<2.0" jaxtyping tyro opencv-python plyfile open3d matplotlib "Pillow>=10" rawpy pytorch-msssim "torchmetrics[image]" typing_extensions tabulate rich \
    && uv pip install --system --no-build-isolation "fused-bilagrid @ git+https://github.com/harry7557558/fused-bilagrid.git@dev" \
    && uv pip install --system --no-build-isolation /opt/spirulae-splat

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
CMD ["sfmapi-spirulae", "--host", "0.0.0.0", "--port", "8080"]
