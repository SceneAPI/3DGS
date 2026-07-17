"""Provider config modules: one manifest + plugin object per radiance provider.

Each submodule carries its provider's ``MANIFEST`` verbatim from the plugin
repo this package supersedes (decision D3), plus the manifest-only
``sfmapi.backends.Plugin`` object that the ``sfmapi.backends`` entry points
expose. Entry-point names (``brush``, ``gsplat``, ``fastergs``, ``lfs``,
``spirulae``) are unchanged from the per-repo packages, so a deployment swaps
packages without config changes.
"""

from __future__ import annotations

from typing import Any

from sfmapi_radiance.providers.brush import MANIFEST as BRUSH_MANIFEST
from sfmapi_radiance.providers.brush import plugin as brush_plugin
from sfmapi_radiance.providers.fastergs import MANIFEST as FASTERGS_MANIFEST
from sfmapi_radiance.providers.fastergs import plugin as fastergs_plugin
from sfmapi_radiance.providers.gsplat import MANIFEST as GSPLAT_MANIFEST
from sfmapi_radiance.providers.gsplat import plugin as gsplat_plugin
from sfmapi_radiance.providers.lfs import MANIFEST as LFS_MANIFEST
from sfmapi_radiance.providers.lfs import plugin as lfs_plugin
from sfmapi_radiance.providers.spirulae import MANIFEST as SPIRULAE_MANIFEST
from sfmapi_radiance.providers.spirulae import plugin as spirulae_plugin

MANIFESTS: dict[str, dict[str, Any]] = {
    "brush": BRUSH_MANIFEST,
    "gsplat": GSPLAT_MANIFEST,
    "fastergs": FASTERGS_MANIFEST,
    "lfs": LFS_MANIFEST,
    "spirulae": SPIRULAE_MANIFEST,
}

PROVIDER_IDS: tuple[str, ...] = tuple(MANIFESTS)

__all__ = [
    "BRUSH_MANIFEST",
    "FASTERGS_MANIFEST",
    "GSPLAT_MANIFEST",
    "LFS_MANIFEST",
    "MANIFESTS",
    "PROVIDER_IDS",
    "SPIRULAE_MANIFEST",
    "brush_plugin",
    "fastergs_plugin",
    "gsplat_plugin",
    "lfs_plugin",
    "spirulae_plugin",
]
