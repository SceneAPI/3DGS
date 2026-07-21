from __future__ import annotations

from typing import Any

MANIFEST: dict[str, Any] = {
    "schema_version": 1,
    "artifact_contracts": [
        "sfmapi.radiance.snapshot.v1",
        "sfmapi.radiance.variant.ply.v1",
        "sfmapi.radiance.metrics.v1",
    ],
    "backend_actions": ["fastergs.*"],
    "capabilities": [
        "radiance.train",
        "radiance.evaluate",
        "radiance.metrics.psnr",
        "radiance.metrics.ssim",
        "radiance.metrics.lpips",
    ],
    "compatibility": {
        "cuda": "required",
        "os": ["windows", "linux"],
        "python": ">=3.12,<3.13",
        "sfmapi": ">=0.0.1",
        "torch": {
            "device": "cuda",
            "index_url": "https://download.pytorch.org/whl/cu128",
            "install_env": {
                "TORCH_DEVICE": "cuda",
                "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
                "TORCH_PACKAGES": "torch torchvision torchaudio",
            },
            "packages": ["torch", "torchvision", "torchaudio"],
            "policy": "required",
        },
    },
    "config_schemas": [
        "radiance.train",
        "fastergs.inference",
        "fastergs.export_ply",
        "fastergs.evaluate",
    ],
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "description": "3D Gaussian Splatting plugin for NeRFICG Faster-GS training, inference, and PLY "
    "export through the sfmapi radiance-field contract.",
    "display_name": "Faster-GS",
    "entry_points": ["gs3.providers.fastergs:plugin"],
    "github_url": "https://github.com/SceneAPI/3DGS.git",
    "licenses": [{"name": "Apache-2.0"}],
    "package_name": "3dgs",
    "plugin_id": "fastergs",
    "providers": [
        {
            "backend_actions": ["fastergs.*"],
            "capabilities": [
                "radiance.train",
                "radiance.evaluate",
                "radiance.metrics.psnr",
                "radiance.metrics.ssim",
                "radiance.metrics.lpips",
            ],
            "display_name": "Faster-GS",
            "priority_hint": 78,
            "provider_id": "fastergs",
        }
    ],
    "runtime_modes": {
        "container_service": {
            "cache": {"path": "/sfmapi/cache", "policy": "read_write", "scope": "plugin"},
            "execution": {
                "artifact_collection": True,
                "env": [
                    "TORCH_HOME",
                    "TORCH_DEVICE",
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES",
                    "NVIDIA_DRIVER_CAPABILITIES",
                ],
                "gpu": "required",
                "log_collection": "both",
                "mounts": {
                    "input_path": "/sfmapi/input",
                    "log_path": "/sfmapi/logs",
                    "output_path": "/sfmapi/output",
                    "work_path": "/sfmapi/work",
                },
                "path": "/execute",
                "retry": {"backoff_seconds": 0, "max_attempts": 1},
                "secrets": [],
                "shutdown_timeout_seconds": 30,
                "timeout_seconds": 86400,
            },
            "healthcheck": {"path": "/healthz", "timeout_seconds": 5},
            "image": {
                "build": {
                    "args": {
                        "SFMAPI_FASTERGS_REF": "main",
                        "TORCH_DEVICE": "cuda",
                        "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
                        "TORCH_PACKAGES": "torch torchvision torchaudio",
                    },
                    "context": "https://github.com/SceneAPI/3DGS.git",
                    "dockerfile": "docker/fastergs.Dockerfile",
                    "ref": "main",
                    "source": "git",
                }
            },
            "object_store": {
                "input_prefix": "fastergs/input/",
                "output_prefix": "fastergs/output/",
                "url_env": "SFMAPI_PLUGIN_OBJECT_STORE_URL",
            },
            "protocol": "sfmapi-plugin-http-v1",
            "protocol_version": "1.1",
            "provenance": {"image_digest_required": False, "source_revision": "main"},
            "service": {
                "default_url": "http://127.0.0.1:8093",
                "url_env": "SFMAPI_FASTERGS_SERVICE_URL",
            },
        },
        "docker": {"build_context": "https://github.com/SceneAPI/3DGS.git", "image": None},
        "external_tool": None,
        "uv": {
            "package": "3dgs",
            "ref": "main",
            "source": "git",
            "url": "https://github.com/SceneAPI/3DGS.git",
        },
    },
    "trust_tier": "community",
    "upstream_projects": [
        {
            "license": None,
            "name": "Faster Gaussian Splatting",
            "url": "https://github.com/nerficg-project/faster-gaussian-splatting",
        }
    ],
}


# FasterGS integrates via container_service rather than registering an
# in-process backend factory, so we use the canonical Plugin's
# manifest-only mode (backend_name + backend_factory default to None
# and register() no-ops). `Plugin` stays exported under the same name.
from sceneapi.backends import Plugin  # noqa: E402

plugin = Plugin(manifest=MANIFEST)

__all__ = ["MANIFEST", "Plugin", "plugin"]
