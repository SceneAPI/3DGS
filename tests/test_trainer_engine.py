"""Unified native-engine tests (brush / lfs / spirulae / fastergs).

The GPU-runtime and ``_normalize_metrics`` cases below existed as identical
copies in each of the four superseded repos; one trainer module means they
are asserted once (parametrized where the behavior is per-provider).
"""

from __future__ import annotations

import subprocess

import pytest

from sceneapi_3dgs import trainer
from sceneapi_3dgs.trainer import PROVIDERS, ExecuteRequest, _normalize_metrics

ENGINE_PROVIDERS = tuple(PROVIDERS)


def test_engine_provider_table_covers_the_four_native_engines() -> None:
    assert set(ENGINE_PROVIDERS) == {"brush", "lfs", "spirulae", "fastergs"}
    assert PROVIDERS["brush"].cuda_required is False
    for provider in ("lfs", "spirulae", "fastergs"):
        assert PROVIDERS[provider].cuda_required is True


def test_train_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match=r"request\.provider must be one of"):
        trainer.train(ExecuteRequest(task_kind="radiance_train", provider="gsplat"))


@pytest.mark.parametrize("provider", ENGINE_PROVIDERS)
def test_gpu_runtime_info_reports_visible_gpu(provider, monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        command = args[0]
        if command[0] == "vulkaninfo":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "GPU0:\n"
                    "    deviceName = NVIDIA Test GPU\n"
                    "    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="NVIDIA Test GPU\n", stderr="")

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    info = trainer._gpu_runtime_info(provider)

    assert info["gpu_runtime_available"] is True
    assert info["gpu_device"] == "NVIDIA Test GPU"


@pytest.mark.parametrize("provider", ENGINE_PROVIDERS)
def test_require_gpu_runtime_rejects_missing_gpu(provider, monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="no gpu")

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GPU runtime is required"):
        trainer._require_gpu_runtime(provider)


def test_normalize_metrics_allows_requested_subset() -> None:
    metrics = _normalize_metrics(
        {"psnr": 21.5, "ssim": 0.82},
        duration_s=1.25,
        required_metrics={"psnr", "ssim"},
    )

    assert metrics is not None
    assert metrics["psnr_db"] == 21.5
    assert metrics["ssim"] == 0.82
    assert metrics["lpips"] is None


def test_normalize_metrics_infers_image_count_from_metric_lists() -> None:
    metrics = _normalize_metrics(
        {
            "avg_psnr": 21.5,
            "avg_ssim_torchmetrics": 0.82,
            "psnr": [20.0, 21.0, 23.5],
            "ssim_torchmetrics": [0.7, 0.8, 0.96],
        },
        duration_s=1.25,
        required_metrics={"psnr", "ssim"},
    )

    assert metrics is not None
    assert metrics["num_images"] == 3


def test_normalize_metrics_rejects_missing_requested_metric() -> None:
    metrics = _normalize_metrics(
        {"psnr": 21.5, "ssim": 0.82},
        duration_s=1.25,
        required_metrics={"psnr", "ssim", "lpips"},
    )

    assert metrics is None
