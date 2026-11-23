"""The version of the installed manim distribution."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

# Use installed distribution version if available; otherwise fall back to a
# sensible default so that importing from a source checkout works without an
# editable install (pip install -e .).
try:
    __version__ = version("manim")
except PackageNotFoundError:
    # Package is not installed; provide a fallback version string.
    __version__ = "0.0.0+unknown"
