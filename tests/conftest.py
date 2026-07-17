from __future__ import annotations

import pytest

from sceneapi_3dgs import gsplat_trainer, trainer
from sceneapi_3dgs.providers import PROVIDER_IDS
from sceneapi_3dgs.server import build_app


def engine_module(provider: str):
    """The trainer module a provider's server dispatches to -- the monkeypatch
    seam the old per-repo suites reached via ``sfmapi_<pkg>.server.train``."""
    return gsplat_trainer if provider == "gsplat" else trainer


@pytest.fixture(params=PROVIDER_IDS)
def provider(request) -> str:
    """Every radiance provider this package serves."""
    return request.param


@pytest.fixture
def app(provider: str):
    return build_app(provider)
