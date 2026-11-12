"""OpenGL rendering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .camera import OpenGLCamera
    from .renderer import OpenGLRenderer

__all__ = ["OpenGLCamera", "OpenGLRenderer"]


def __getattr__(name: str) -> Any:
    value: Any
    if name == "OpenGLCamera":
        from .camera import OpenGLCamera

        value = OpenGLCamera
    elif name == "OpenGLRenderer":
        from .renderer import OpenGLRenderer

        value = OpenGLRenderer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
