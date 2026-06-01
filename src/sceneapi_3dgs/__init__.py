"""SceneAPI 3DGS plugin package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("3DGS")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__"]
