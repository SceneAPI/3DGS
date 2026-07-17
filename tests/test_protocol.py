"""Kit-shape protocol tests, shared across all five providers.

The five superseded repos carried per-repo copies of these tests (identical
modulo module names); here they run parametrized over every provider via the
``provider`` / ``app`` fixtures. The wrong-provider and radiance_eval
dispatch cases originated in the gsplat suite and now hold for all five.
"""

from __future__ import annotations

from conftest import engine_module
from fastapi.testclient import TestClient
from sceneapi.plugin_service import PROTOCOL, PROTOCOL_VERSION

from sceneapi_3dgs.providers import MANIFESTS
from sceneapi_3dgs.trainer import ExecuteRequest


def test_server_speaks_kit_protocol_1_1(provider, app) -> None:
    client = TestClient(app)

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    version_response = client.get("/version")
    assert version_response.status_code == 200
    version = version_response.json()
    assert version["protocol"] == PROTOCOL
    assert version["protocol_version"] == PROTOCOL_VERSION == "1.1"
    assert version["plugin_id"] == provider
    assert version["runtime"]["provider"] == provider
    assert (
        MANIFESTS[provider]["runtime_modes"]["container_service"]["protocol_version"]
        == PROTOCOL_VERSION
    )


def test_capabilities_serves_the_manifest_capability_set(provider, app) -> None:
    features = TestClient(app).get("/capabilities").json()["features"]

    assert features == sorted(MANIFESTS[provider]["capabilities"])


def test_execute_routes_train_requests_to_the_trainer(provider, app, monkeypatch) -> None:
    captured: dict[str, ExecuteRequest] = {}

    def fake_train(request: ExecuteRequest) -> dict[str, object]:
        captured["request"] = request
        return {"snapshot": {"seq": 1}}

    monkeypatch.setattr(engine_module(provider), "train", fake_train)

    response = TestClient(app).post(
        "/execute",
        json={
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "task_kind": "radiance_train",
            "capability": "radiance.train",
            "provider": provider,
            "inputs": {"project_id": "p", "radiance_field_id": "r"},
            "spec": {"backend_options": {"dataset_path": "/tmp/missing"}},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "protocol": PROTOCOL,
        "status": "succeeded",
        "outputs": {"snapshot": {"seq": 1}},
    }
    request = captured["request"]
    assert request.provider == provider
    assert request.inputs == {"project_id": "p", "radiance_field_id": "r"}
    assert request.spec == {"backend_options": {"dataset_path": "/tmp/missing"}}


def test_execute_maps_trainer_errors_to_failed_status(provider, app, monkeypatch) -> None:
    def boom(request: ExecuteRequest) -> dict[str, object]:
        raise RuntimeError("GPU runtime is required")

    monkeypatch.setattr(engine_module(provider), "train", boom)

    response = TestClient(app).post(
        "/execute",
        json={
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "task_kind": "radiance_train",
            "provider": provider,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "RuntimeError: GPU runtime is required" in body["error"]


def test_execute_rejects_wrong_protocol(provider, app) -> None:
    response = TestClient(app).post("/execute", json={"protocol": "nope", "task_kind": "x"})

    assert response.status_code == 400
    assert response.json()["error"] == "protocol_mismatch"


def test_execute_rejects_wrong_provider(provider, app) -> None:
    response = TestClient(app).post(
        "/execute",
        json={
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "task_kind": "radiance_train",
            "capability": "radiance.train",
            "provider": "other",
            "inputs": {},
            "spec": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert f"request.provider must be '{provider}'" in body["error"]


def test_execute_dispatches_radiance_eval(provider, app, monkeypatch) -> None:
    def fake_evaluate(request: ExecuteRequest) -> dict[str, object]:
        return {
            "radiance_field_id": request.inputs["radiance_field_id"],
            "evaluation_id": request.inputs["evaluation_id"],
            "snapshot_seq": 1,
            "metrics": {
                "psnr_db": 30.0,
                "ssim": 1.0,
                "lpips": 0.0,
                "num_images": 1,
                "duration_s": 0.0,
                "render_time_s_total": 0.0,
                "render_time_s_mean": 0.0,
            },
            "artifacts": [],
        }

    monkeypatch.setattr(engine_module(provider), "evaluate", fake_evaluate)

    response = TestClient(app).post(
        "/execute",
        json={
            "protocol": PROTOCOL,
            "protocol_version": PROTOCOL_VERSION,
            "task_kind": "radiance_eval",
            "capability": "radiance.evaluate",
            "provider": provider,
            "inputs": {
                "project_id": "p",
                "radiance_field_id": "rf",
                "evaluation_id": "ev",
            },
            "spec": {"snapshot_seq": 1},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["outputs"]["evaluation_id"] == "ev"
    assert body["outputs"]["metrics"]["psnr_db"] == 30.0
