"""OpenGL rendering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .renderer import OpenGLCamera, OpenGLRenderer

__all__ = ["OpenGLCamera", "OpenGLRenderer"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .renderer import OpenGLCamera, OpenGLRenderer

    value = {"OpenGLCamera": OpenGLCamera, "OpenGLRenderer": OpenGLRenderer}[name]
    globals()[name] = value
    return value
