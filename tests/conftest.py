from __future__ import annotations

import pytest

from gs3 import gsplat_trainer, trainer
from gs3.providers import PROVIDER_IDS
from gs3.server import build_app


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
