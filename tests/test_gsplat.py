"""gsplat-specific behavior (its trainer is the genuinely different one),
ported from sfmapi_gsplat/tests/test_protocol.py. The kit-shape cases from
that suite (health/version, capabilities, wrong protocol, wrong provider,
radiance_eval dispatch) live in the shared parametrized test_protocol.py."""

from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient
from sfmapi.plugin_service import PROTOCOL, PROTOCOL_VERSION

from sfmapi_radiance import gsplat_trainer as trainer
from sfmapi_radiance.server import build_app


def test_gpu_runtime_info_reports_visible_gpu(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="NVIDIA Test GPU\n", stderr="")

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    info = trainer._gpu_runtime_info()

    assert info["gpu_runtime_available"] is True
    assert info["gpu_device"] == "NVIDIA Test GPU"


def test_require_gpu_runtime_rejects_missing_gpu(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="no gpu")

    monkeypatch.setattr(trainer.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GPU runtime is required"):
        trainer._require_gpu_runtime()


def test_execute_surfaces_missing_cuda_as_plugin_failure(monkeypatch) -> None:
    def fake_train(_request):
        raise RuntimeError("CUDA is required for sfmapi-gsplat training")

    monkeypatch.setattr(trainer, "train", fake_train)
    client = TestClient(build_app("gsplat"))

    response = client.post(
        "/execute",
        json={
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "task_kind": "radiance_train",
            "capability": "radiance.train",
            "provider": "gsplat",
            "inputs": {"project_id": "p", "radiance_field_id": "rf"},
            "spec": {"method": "gsplat.train.default", "max_steps": 1},
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "CUDA is required" in response.json()["error"]


def test_normalized_metrics_keep_single_image_smoke_scope() -> None:
    metrics = trainer._normalize_metrics_payload(
        {"psnr_db": 30.0, "ssim": 0.9, "lpips": 0.1},
        duration_s=0.25,
    )

    assert metrics["num_images"] == 1
    assert metrics["eval_protocol"] == "single_image_smoke"
