"""OpenGL rendering backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from manim.renderer.opengl.camera import OpenGLCamera
    from manim.renderer.opengl.renderer import OpenGLRenderer

__all__ = ["OpenGLCamera", "OpenGLRenderer"]


def __getattr__(name: str) -> Any:
    value: Any
    if name == "OpenGLCamera":
        from manim.renderer.opengl.camera import OpenGLCamera

        value = OpenGLCamera
    elif name == "OpenGLRenderer":
        from manim.renderer.opengl.renderer import OpenGLRenderer

        value = OpenGLRenderer
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    globals()[name] = value
    return value
