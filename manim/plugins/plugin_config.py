from __future__ import annotations

from pydantic import BaseModel

from manim.renderer.opengl_renderer import OpenGLRenderer
from manim.renderer.renderer import RendererProtocol

__all__ = ("plugins",)


class PluginConfig(BaseModel):
    class Config:
        # runtime check Protocols (must be runtime_checkable Protocols)
        allow_arbitrary_types = True
        # validate setting attributes
        validate_assignment = True

    renderer: type[RendererProtocol]


plugins = PluginConfig(
    renderer=OpenGLRenderer,
)
