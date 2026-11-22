"""Renderer selection: the factories of the built-in renderers.

A factory is a renderer selection with its options. It is immutable and knows what
its renderers can do without creating one; :meth:`create` makes a renderer (each
scene gets its own). A scene is given a factory, or the name of a built-in renderer::

    Scene(renderer="opengl")
    Scene(renderer=OpenGL(anti_alias_width=2))

This module is composition code: it names the renderer packages and imports them
only when a renderer is created.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from manim.constants import RendererType
from manim.renderer.protocol import (
    PreviewRequest,
    Renderer,
    RendererCapabilities,
    RendererFactory,
)

if TYPE_CHECKING:
    from manim._config.utils import ManimConfig

__all__ = ["Cairo", "OpenGL", "backend"]


@dataclass(frozen=True)
class Cairo:
    """The Cairo renderer: draws on the CPU, has no preview window."""

    @property
    def name(self) -> str:
        return "cairo"

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(live_preview=False)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {}

    def create(self, preview: PreviewRequest | None = None) -> Renderer:
        if preview is not None:
            raise ValueError("The Cairo renderer has no preview window.")
        from manim.renderer.cairo import CairoRenderer

        return CairoRenderer()


@dataclass(frozen=True)
class OpenGL:
    """The OpenGL renderer, which can show frames in a preview window.

    Parameters
    ----------
    wireframe
        Draw the edges of triangles only, for debugging.
    anti_alias_width
        The width of the antialiased edge of shapes, in pixels.
    """

    wireframe: bool = False
    anti_alias_width: float = 1.5

    @property
    def name(self) -> str:
        return "opengl"

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(live_preview=True)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {"wireframe": self.wireframe, "anti_alias_width": self.anti_alias_width}

    def create(self, preview: PreviewRequest | None = None) -> Renderer:
        from manim._config import config
        from manim.renderer.opengl.renderer import OpenGLRenderer
        from manim.renderer.opengl.window_settings import _WindowSettings

        renderer = OpenGLRenderer(
            wireframe=self.wireframe, anti_alias_width=self.anti_alias_width
        )
        if preview is not None:
            renderer._request_preview(preview, _WindowSettings.from_config(config))
        return renderer


def backend(
    selection: str | RendererType | RendererFactory | None, config: ManimConfig
) -> RendererFactory:
    """The factory a scene's ``renderer`` argument selects.

    ``None`` selects ``config.renderer``; the name of a built-in renderer selects
    its factory with the options the configuration has for it; a factory selects
    itself.

    Raises
    ------
    TypeError
        If ``selection`` is a renderer instance or something else.
    """
    if selection is None:
        selection = config.renderer
    if isinstance(selection, str) and not isinstance(selection, RendererType):
        selection = RendererType(selection.lower())
    if selection is RendererType.CAIRO:
        return Cairo()
    if selection is RendererType.OPENGL:
        return OpenGL(wireframe=bool(config.enable_wireframe))
    if isinstance(selection, RendererFactory):
        return selection
    if isinstance(selection, Renderer):
        raise TypeError(
            f"Scene(renderer=...) takes a renderer factory or name, not a renderer "
            f"({type(selection).__name__}); pass for example "
            "manim.renderer.backends.Cairo() or 'opengl'. Each scene creates its "
            "own renderer."
        )
    raise TypeError(
        f"Scene(renderer=...) takes a renderer factory or name, not {selection!r}."
    )
