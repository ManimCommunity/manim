"""A renderer written against the renderer interface only.

It imports nothing of manim but :mod:`manim.renderer.protocol`,
:mod:`manim.renderer.snapshot` and numpy, and draws the bounding box of every path,
point cloud and image, in one colour per node, into a :class:`.PixelFrame`. Its one
option (``shade``) changes its pixels. It has no cache and no private hooks: the
Manager, the play cache, images and evaluation must work with it as they do with
the built-in renderers.

It is also the example of the "Writing a renderer" guide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from manim.renderer.protocol import (
    PixelFrame,
    PreviewRequest,
    RenderConfig,
    Renderer,
    RendererCapabilities,
    RendererFactory,
    RendererState,
)
from manim.renderer.snapshot import RNode

__all__ = ["Plain", "PlainRenderer"]


class PlainRenderer(Renderer):
    """Draws bounding boxes, ``shade`` bright, over the frame's background."""

    def __init__(self, shade: int) -> None:
        self.shade = shade
        self._state = RendererState.UNOPENED
        #: Every frame drawn, as (number of nodes, size), for tests.
        self.drawn: list[tuple[int, tuple[int, int]]] = []

    @property
    def state(self) -> RendererState:
        return self._state

    def render(self, nodes: Sequence[RNode], config: RenderConfig, /) -> PixelFrame:
        if not self._state.accepts_work:
            raise RuntimeError(f"The plain renderer is {self._state.value}.")

        self._state = RendererState.OPEN
        width, height = config.pixel_width, config.pixel_height
        camera = config.camera
        scale = np.array([width / camera.frame_width, height / camera.frame_height])

        # Unlike the guide's version, it starts from the frame's background.
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        background = np.round(
            np.array([*config.background_color.to_rgb(), config.background_opacity])
            * 255
        )
        pixels[:] = background.astype(np.uint8)
        for node in nodes:
            points = node.payload.points
            if points.ndim != 2 or len(points) == 0:
                continue
            # the bounding box of the node's points, in pixels
            xy = (points[:, :2] - np.asarray(camera.center)[:2]) * scale
            low = np.floor(xy.min(axis=0) + [width / 2, height / 2]).astype(int)
            high = np.ceil(xy.max(axis=0) + [width / 2, height / 2]).astype(int)
            # clipped to the frame; row zero is the top
            x0, x1 = np.clip([low[0], high[0]], 0, width)
            y0, y1 = np.clip([height - high[1], height - low[1]], 0, height)
            pixels[y0:y1, x0:x1] = [self.shade, self.shade, self.shade, 255]
        self.drawn.append((len(nodes), (width, height)))
        return PixelFrame(pixels)

    def close(self) -> None:
        self._state = RendererState.CLOSED


@dataclass(frozen=True)
class Plain(RendererFactory):
    """The factory of :class:`PlainRenderer`; ``shade`` changes the pixels."""

    shade: int = 255

    @property
    def name(self) -> str:
        return "plain"

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(live_preview=False)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {"shade": self.shade}

    def create(self, preview: PreviewRequest | None = None) -> PlainRenderer:
        if preview is not None:
            raise ValueError("The plain renderer has no preview window.")
        return PlainRenderer(self.shade)


@dataclass(frozen=True, eq=False)
class Given:
    """A factory whose renderer is given (a mock, in tests of the Manager)."""

    renderer: Any

    @property
    def name(self) -> str:
        return "given"

    @property
    def capabilities(self) -> RendererCapabilities:
        return RendererCapabilities(live_preview=False)

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        return {}

    def create(self, preview: PreviewRequest | None = None) -> Any:
        return self.renderer
