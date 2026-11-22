"""What a renderer is given to draw a frame, its capabilities and lifecycle state,
and the drawing methods used by Manager.
"""

from __future__ import annotations

__all__ = [
    "CameraState",
    "CapabilityError",
    "RenderConfig",
    "Renderer",
    "RendererCapabilities",
    "RendererState",
]

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from manim.scene.scene import Scene
    from manim.typing import Point3D, RGBAPixelArray
    from manim.utils.color import ManimColor


class CapabilityError(Exception):
    """A renderer was asked to draw something it cannot draw.

    Raised instead of drawing it differently, so that no renderer silently shows
    something else than another.
    """


@dataclass(frozen=True, slots=True, eq=False)
class CameraState:
    """A camera at one time, as plain values: what renderers draw from.

    Made by :meth:`.Camera._state`; renderers never read camera objects.
    """

    center: Point3D
    """The center of the camera's frame in scene coordinates."""
    frame_width: float
    """The width of the frame in scene units."""
    frame_height: float
    """The height of the frame in scene units."""
    rotation: np.ndarray
    """The rotation from scene to view coordinates (3x3); the view looks along -z."""
    focal_distance: float | None
    """The distance of the eye from the frame center, for the perspective of a
    :class:`.ThreeDCamera`; ``None`` for a flat view, which orders nothing by
    depth."""
    zoom: float
    """The factor that projected points are scaled by (1 for a flat view)."""
    light: Point3D | None
    """Where the camera's light is, in scene coordinates; ``None`` for no light
    (nothing is lit). Provisional: lights become scene objects."""
    shade_3d: bool
    """Whether ``shade_in_3d`` paths are shaded towards :attr:`light`.
    Provisional, as that shading."""
    default_frame_width: float
    """The frame width the scene was configured with: mobjects fixed in frame are
    drawn as if the frame had this width and were centered at the origin."""


@dataclass(frozen=True, slots=True, eq=False)
class RenderConfig:
    """What a renderer is given next to the render snapshots of one frame."""

    pixel_width: int
    pixel_height: int
    background_color: ManimColor
    background_opacity: float
    camera: CameraState
    background_image: str | None = None
    """An image file shown behind everything, scaled to the frame (Cairo only)."""


class RendererState(enum.Enum):
    """Where a renderer is in its life, as its read-only ``state`` property says.

    Every renderer has a ``state``. A renderer is created ``UNOPENED`` and
    acquires its drawing resources when it is opened (explicitly, or when it
    first needs them): ``OPEN``. :meth:`close` releases them: ``CLOSING`` while
    it does, then ``CLOSED``; if releasing fails while the resources still
    exist, the renderer is ``CLOSE_FAILED`` and ``close()`` can be called again.
    A renderer that was never opened closes directly.

    Ask the derived properties, such as :attr:`accepts_work`, rather than
    comparing with sets of states, so that a new state changes the answer in
    one place.
    """

    UNOPENED = "unopened"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    CLOSE_FAILED = "close failed"

    @property
    def accepts_work(self) -> bool:
        """Whether the renderer can be opened, draw, or be given a scene."""
        return self in (RendererState.UNOPENED, RendererState.OPEN)


@dataclass(frozen=True, slots=True)
class RendererCapabilities:
    """Optional session features implemented by a renderer."""

    live_preview: bool = False


@runtime_checkable
class Renderer(Protocol):
    """What the Manager requires of every renderer, beyond drawing.

    A renderer declares its optional features and reports where it is in its
    life; the Manager opens it before a run and closes it afterwards, and reads
    nothing else of it.
    """

    capabilities: ClassVar[RendererCapabilities]

    @property
    def state(self) -> RendererState:
        """Where the renderer is in its life."""
        ...

    def open(self) -> None:
        """Acquire the drawing resources, if not done yet."""
        ...

    def close(self) -> None:
        """Release the drawing resources; the renderer does not draw afterwards."""
        ...


class _AnimationRenderer(Protocol):
    """Draw frames and display them in a preview window at Manager's request.

    Manager prepares animations, checks the cache, advances animation time, and
    sends frames to the writer. The renderer prepares its drawing resources and
    uses wall-clock time to pace its live preview.
    """

    def _animation_cache_identity(self, scene: Scene) -> tuple[str, Any]: ...

    def _start_animation(self) -> None: ...

    def render(self, scene: Scene, frame_offset: float, /) -> None: ...

    def get_frame(self) -> RGBAPixelArray: ...

    def _present_frame(self, scene: Scene, frame_offset: float) -> None: ...

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None: ...
