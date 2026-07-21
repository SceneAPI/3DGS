"""One kit-based ``sfmapi-plugin-http-v1`` server, parameterized by provider.

``build_app(provider)`` builds the ASGI app that the old per-repo
``sfmapi_<pkg>.server:app`` module attribute used to be: it serves that
provider's manifest through the ``sceneapi.plugin_service`` kit and dispatches
``/execute`` tasks to the matching trainer engine. The per-provider
``sfmapi-<provider>`` console scripts keep their old names and defaults.
"""

from __future__ import annotations

import argparse
import traceback
from typing import Any

from sceneapi.plugin_service import ManifestBackend, TaskExecutor, build_plugin_server

from gs3 import __version__
from gs3.providers import MANIFESTS
from gs3.trainer import ExecuteRequest


def _engine(provider: str) -> Any:
    """Resolve the trainer engine module for ``provider`` at call time.

    gsplat's engine imports numpy/pillow at module import, so it is resolved
    lazily: launching (or testing) any other provider must not require the
    gsplat extras. Call-time resolution also keeps the monkeypatch seam the
    old per-repo suites used (``sfmapi_<pkg>.server.train``): patching
    ``gs3.trainer.train`` or
    ``gs3.gsplat_trainer.train`` reroutes dispatch immediately.
    """
    if provider == "gsplat":
        from gs3 import gsplat_trainer

        return gsplat_trainer
    from gs3 import trainer

    return trainer


def _make_executor(plugin_provider: str) -> TaskExecutor:
    """Kit executor for one provider: dispatch each task to its trainer,
    mapping trainer errors onto the ``status: failed`` result the sceneapi
    worker expects."""

    def execute_task(
        *,
        task_kind: str,
        capability: str,
        inputs: dict[str, Any],
        spec: dict[str, Any],
        tenant_id: str,
        job_id: str,
        task_id: str,
        provider: str,
    ) -> dict[str, Any]:
        request = ExecuteRequest(
            task_kind=task_kind,
            capability=capability,
            inputs=inputs,
            spec=spec,
            tenant_id=tenant_id,
            job_id=job_id,
            task_id=task_id,
            provider=provider,
        )
        try:
            if provider != plugin_provider:
                raise ValueError(f"request.provider must be {plugin_provider!r}")
            engine = _engine(plugin_provider)
            outputs = (
                engine.evaluate(request) if task_kind == "radiance_eval" else engine.train(request)
            )
        except Exception as exc:
            return {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=20)}",
            }
        return {"status": "succeeded", "outputs": outputs}

    return execute_task


def _make_runtime_info(provider: str) -> Any:
    def _runtime_info() -> dict[str, Any]:
        engine = _engine(provider)
        if provider == "gsplat":
            return engine.runtime_info()
        return engine.runtime_info(provider)

    return _runtime_info


def build_app(provider: str) -> Any:
    manifest = MANIFESTS[provider]
    return build_plugin_server(
        ManifestBackend(manifest, version=__version__),
        plugin_id=manifest["plugin_id"],
        package_version=__version__,
        executor=_make_executor(provider),
        runtime_info=_make_runtime_info(provider),
    )


def main(provider: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    import uvicorn

    uvicorn.run(build_app(provider), host=args.host, port=args.port)
    return 0


def brush_main(argv: list[str] | None = None) -> int:
    return main("brush", argv)


def gsplat_main(argv: list[str] | None = None) -> int:
    return main("gsplat", argv)


def fastergs_main(argv: list[str] | None = None) -> int:
    return main("fastergs", argv)


def lfs_main(argv: list[str] | None = None) -> int:
    return main("lfs", argv)


def spirulae_main(argv: list[str] | None = None) -> int:
    return main("spirulae", argv)
