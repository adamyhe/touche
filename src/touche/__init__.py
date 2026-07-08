"""touche: chromatin contact preprocessing and analysis tools."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("ep-touche")
except PackageNotFoundError:
    __version__ = "unknown"
