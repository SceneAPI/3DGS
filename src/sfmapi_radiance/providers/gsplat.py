from __future__ import annotations

from typing import Any

MANIFEST: dict[str, Any] = {
    "schema_version": 1,
    "plugin_id": "gsplat",
    "display_name": "gsplat",
    "description": (
        "3D Gaussian Splatting plugin for gsplat training, evaluation, and PLY "
        "snapshot export through the sfmapi radiance-field contract."
    ),
    "package_name": "sfmapi-gsplat",
    "github_url": "https://github.com/SFMAPI/sfmapi_gsplat.git",
    "entry_points": ["sfmapi_gsplat.plugin:plugin"],
    "providers": [
        {
            "provider_id": "gsplat",
            "display_name": "gsplat",
            "capabilities": [
                "radiance.train",
                "radiance.evaluate",
                "radiance.metrics.psnr",
                "radiance.metrics.ssim",
                "radiance.metrics.lpips",
            ],
            "backend_actions": ["gsplat.*"],
            "priority_hint": 70,
        }
    ],
    "runtime_modes": {
        "uv": {
            "source": "git",
            "url": "https://github.com/SFMAPI/sfmapi_gsplat.git",
            "ref": "main",
            "package": "sfmapi-gsplat",
        },
        "docker": {
            "image": None,
            "build_context": "https://github.com/SFMAPI/sfmapi_gsplat.git",
        },
        "container_service": {
            "protocol": "sfmapi-plugin-http-v1",
            "protocol_version": "1.1",
            "service": {
                "default_url": "http://127.0.0.1:8098",
                "url_env": "SFMAPI_GSPLAT_SERVICE_URL",
            },
            "image": {
                "build": {
                    "source": "git",
                    "context": "https://github.com/SFMAPI/sfmapi_gsplat.git",
                    "dockerfile": "Dockerfile",
                    "ref": "main",
                    "args": {
                        "GSPLAT_PACKAGE": "gsplat==1.5.3",
                        "TORCH_CUDA_ARCH_LIST": "12.0",
                    },
                }
            },
            "object_store": {
                "url_env": "SFMAPI_PLUGIN_OBJECT_STORE_URL",
                "input_prefix": "gsplat/input/",
                "output_prefix": "gsplat/output/",
            },
            "cache": {"policy": "read_write", "scope": "plugin", "path": "/sfmapi/cache"},
            "provenance": {"image_digest_required": False, "source_revision": "main"},
            "healthcheck": {"path": "/healthz", "timeout_seconds": 5},
            "execution": {
                "path": "/execute",
                "timeout_seconds": 86400,
                "mounts": {
                    "input_path": "/sfmapi/input",
                    "output_path": "/sfmapi/output",
                    "work_path": "/sfmapi/work",
                    "log_path": "/sfmapi/logs",
                },
                "gpu": "required",
                "env": [
                    "TORCH_HOME",
                    "TORCH_DEVICE",
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES",
                    "NVIDIA_DRIVER_CAPABILITIES",
                ],
                "secrets": [],
                "retry": {"max_attempts": 1, "backoff_seconds": 0},
                "shutdown_timeout_seconds": 30,
                "log_collection": "both",
                "artifact_collection": True,
            },
        },
        "external_tool": None,
    },
    "capabilities": [
        "radiance.train",
        "radiance.evaluate",
        "radiance.metrics.psnr",
        "radiance.metrics.ssim",
        "radiance.metrics.lpips",
    ],
    "backend_actions": ["gsplat.*"],
    "config_schemas": ["radiance.train"],
    "artifact_contracts": [
        "sfmapi.radiance.snapshot.v1",
        "sfmapi.radiance.metrics.v1",
        "sfmapi.radiance.variant.ply.v1",
    ],
    "licenses": [{"name": "Apache-2.0"}],
    "upstream_projects": [
        {
            "name": "gsplat",
            "url": "https://github.com/nerfstudio-project/gsplat",
            "license": "Apache-2.0",
        }
    ],
    "compatibility": {
        "sfmapi": ">=0.0.1",
        "python": ">=3.10,<3.13",
        "os": ["windows", "linux"],
        "cuda": "required",
        "torch": {
            "policy": "required",
            "device": "cuda",
            "index_url": "https://download.pytorch.org/whl/cu128",
            "packages": ["torch"],
            "install_env": {
                "TORCH_DEVICE": "cuda",
                "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
                "GSPLAT_PACKAGE": "gsplat==1.5.3",
                "TORCH_CUDA_ARCH_LIST": "12.0",
            },
        },
        "tool_versions": {
            "cuda": "12.8",
            "torch": "2.7.1+cu128",
            "gsplat": "1.5.3+source",
            "pycolmap": ">=3.11",
        },
    },
    "conformance": {"status": "not_run", "suite": "sfmapi-bench"},
    "trust_tier": "community",
}


# gsplat integrates via container_service rather than registering an
# in-process backend factory, so we use the canonical Plugin's
# manifest-only mode (backend_name + backend_factory default to None
# and register() no-ops). `Plugin` stays exported under the same name.
from sfmapi.backends import Plugin  # noqa: E402

plugin = Plugin(manifest=MANIFEST)
