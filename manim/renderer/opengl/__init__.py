"""OpenGL rendering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .renderer import OpenGLRenderer

__all__ = ["OpenGLRenderer"]


def __getattr__(name: str) -> Any:
    value: Any
    if name == "OpenGLRenderer":
        from .renderer import OpenGLRenderer

        value = OpenGLRenderer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
