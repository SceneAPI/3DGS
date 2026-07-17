"""Brush-specific engine behavior (Vulkan runtime detection + native log
metrics), ported from sfmapi_brush/tests/test_protocol.py."""

from __future__ import annotations

import subprocess

from sceneapi_3dgs import trainer
from sceneapi_3dgs.trainer import _parse_brush_metrics


def test_vulkan_gpu_runtime_accepts_hardware_when_software_device_is_also_listed(
    monkeypatch,
) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "GPU0:\n"
                "    deviceName = NVIDIA Test GPU\n"
                "    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU\n"
                "GPU1:\n"
                "    deviceName = llvmpipe\n"
                "    deviceType = PHYSICAL_DEVICE_TYPE_CPU\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    info = trainer._vulkan_gpu_runtime_info()

    assert info["gpu_runtime_available"] is True
    assert info["gpu_devices"] == ["NVIDIA Test GPU"]


def test_vulkan_gpu_runtime_rejects_software_only(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            stdout=(
                "GPU0:\n    deviceName = llvmpipe\n    deviceType = PHYSICAL_DEVICE_TYPE_CPU\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    info = trainer._vulkan_gpu_runtime_info()

    assert info["gpu_runtime_available"] is False
    assert "software or CPU" in info["gpu_error"]


def test_parse_brush_eval_log() -> None:
    process = {"stdout": "Eval iter 7: PSNR 18.25, ssim 0.711\n", "stderr": ""}

    metrics = _parse_brush_metrics(process, duration_s=None)

    assert metrics is not None
    assert metrics["iteration"] == 7
    assert metrics["psnr_db"] == 18.25
    assert metrics["ssim"] == 0.711


def test_count_eval_split_images_prefers_image_folder(tmp_path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for idx in range(10):
        (images / f"{idx}.jpg").write_bytes(b"fake")

    assert trainer._count_eval_split_images(tmp_path, 4) == 3
