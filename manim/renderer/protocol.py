"""What a renderer is given to draw a frame, its capabilities and lifecycle state,
and the drawing methods used by Manager.
"""

from __future__ import annotations

__all__ = [
    "CameraState",
    "CameraView",
    "CapabilityError",
    "Frame",
    "LightState",
    "PixelFrame",
    "Preview",
    "PreviewInput",
    "PreviewRenderer",
    "PreviewRequest",
    "RenderConfig",
    "Renderer",
    "RendererCapabilities",
    "RendererFactory",
    "RendererState",
]

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

    import numpy as np

    from manim.renderer.snapshot import RNode
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
    default_frame_width: float
    """The frame width the scene was configured with: mobjects fixed in frame are
    drawn as if the frame had this width and were centered at the origin."""


@dataclass(frozen=True, slots=True, eq=False)
class LightState:
    """A point light at one time (a :class:`~.PointLight`)."""

    position: Point3D
    """Where the light is, in scene coordinates."""
    rgb: tuple[float, float, float]
    """Its color, linear RGB."""
    intensity: float
    """Its intensity in candela (glTF ``KHR_lights_punctual``)."""


@dataclass(frozen=True, slots=True, eq=False)
class CameraView:
    """What one camera view (an :class:`.ImageMobjectFromCamera`) shows in a frame.

    The renderer draws :attr:`nodes` as :attr:`config` sets them up, and shows the
    picture on the frame's node with :attr:`uid` (an :class:`.RCameraView`), and on
    the copies of that node, which have the same :attr:`camera_id`.
    """

    uid: int
    camera_id: int
    config: RenderConfig
    nodes: Sequence[RNode]


@dataclass(frozen=True, slots=True, eq=False)
class RenderConfig:
    """What a renderer is given next to the render snapshots of one frame."""

    pixel_width: int
    pixel_height: int
    background_color: ManimColor
    background_opacity: float
    camera: CameraState
    reference_pixel_size: tuple[int, int]
    """The pixel size of the scene's frames. Widths given in pixels, such as the
    thickness of point clouds, are meant at this size: a camera view, which has
    fewer pixels, draws them thinner, as they appear in the frame."""
    background_image: str | None = None
    """An image file shown behind everything, scaled to the frame (Cairo only)."""
    camera_views: tuple[CameraView, ...] = ()
    """The camera views of the frame, drawn before it. They have no views."""
    lights: tuple[LightState, ...] = ()
    """The scene's lights. Without lights, nothing is shaded: lit materials show
    their base color."""
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The light that reaches every surface everywhere, linear RGB."""


@runtime_checkable
class Frame(Protocol):
    """A frame a renderer drew, where the renderer keeps it (for example on the
    GPU) until someone asks for its pixels.

    A preview shows a frame without reading it back; the pixels are read the first
    time :meth:`pixels` is called. :meth:`pixels` can be called until the renderer
    draws its next frame or closes; the array it returns belongs to the caller and
    stays valid and unchanged for as long as the caller keeps it.
    """

    @property
    def size(self) -> tuple[int, int]:
        """The width and height of the frame in pixels."""
        ...

    def pixels(self) -> RGBAPixelArray:
        """The frame as RGBA pixels, row zero at the top; the same array on every
        call.
        """
        ...


class PixelFrame:
    """A :class:`Frame` whose pixels are already in memory, owned by the frame."""

    __slots__ = ("_pixels",)

    def __init__(self, pixels: RGBAPixelArray) -> None:
        self._pixels = pixels

    @property
    def size(self) -> tuple[int, int]:
        height, width = self._pixels.shape[:2]
        return width, height

    def pixels(self) -> RGBAPixelArray:
        return self._pixels


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
    """Draws frames: render snapshots and a :class:`RenderConfig` in, a
    :class:`Frame` out.

    A renderer never reads a scene, a mobject or a camera: the Manager's
    :class:`.FrameBuilder` makes what it is given. It may keep what it built
    between frames, keyed by :attr:`.RNode.key`. It acquires its resources when it
    first needs them, and reports where it is in its life; the Manager closes it.
    Renderers are made by a :class:`RendererFactory`, one per scene.
    """

    @property
    def state(self) -> RendererState:
        """Where the renderer is in its life."""
        ...

    def render(self, nodes: Sequence[RNode], config: RenderConfig, /) -> Frame:
        """Draw ``nodes`` in order, as ``config`` sets them up; return the frame."""
        ...

    def close(self) -> None:
        """Release the drawing resources; the renderer does not draw afterwards."""
        ...


class PreviewInput(Protocol):
    """What a preview window sends its input to (a :class:`.Scene` is one): mouse
    positions and offsets as scene points, keys as pyglet key codes.
    """

    def on_mouse_motion(self, point: Point3D, d_point: Point3D) -> None: ...

    def on_mouse_scroll(self, point: Point3D, offset: Point3D) -> None: ...

    def on_key_press(self, symbol: int, modifiers: int) -> None: ...

    def on_key_release(self, symbol: int, modifiers: int) -> None: ...

    def on_mouse_drag(
        self, point: Point3D, d_point: Point3D, buttons: int, modifiers: int
    ) -> None: ...

    def on_mouse_press(self, point: Point3D, button: str, modifiers: int) -> None: ...


@dataclass(frozen=True, slots=True)
class PreviewRequest:
    """A request for a preview window, given to a renderer when it is created."""

    input: PreviewInput
    """Where the window sends its input."""
    view: Callable[[], CameraState]
    """The camera state window positions are mapped to scene points with."""


@runtime_checkable
class Preview(Protocol):
    """The handle of a requested preview window.

    The window opens on :meth:`show` or with the renderer's first frame,
    whichever comes first; the renderer's ``close()`` closes it.
    """

    @property
    def is_closing(self) -> bool:
        """Whether the user asked the window to close."""
        ...

    @property
    def pressed_keys(self) -> Collection[int]:
        """The keys held down in the window now."""
        ...

    def show(self) -> None:
        """Open the window if it is not open yet."""
        ...

    def present(self, frame: Frame) -> None:
        """Show ``frame``, a frame of the renderer, in the window."""
        ...


@runtime_checkable
class PreviewRenderer(Renderer, Protocol):
    """A renderer that can show its frames in a preview window."""

    @property
    def preview(self) -> Preview | None:
        """The preview handle; ``None`` if no preview was requested."""
        ...


@runtime_checkable
class RendererFactory(Protocol):
    """A renderer selection with its options; immutable.

    A scene is given a factory (``Scene(renderer=factory)``) and creates its
    renderer with it; :class:`.Manager` creates a second one for images apart from
    the frames of a run.
    """

    @property
    def name(self) -> str:
        """The renderer's name, also the backend part of play-cache keys."""
        ...

    @property
    def capabilities(self) -> RendererCapabilities:
        """What the factory's renderers can do, known without creating one."""
        ...

    @property
    def cache_identity(self) -> Mapping[str, Any]:
        """The options that change the pixels, for play-cache keys."""
        ...

    def create(self, preview: PreviewRequest | None = None) -> Renderer:
        """A new renderer; with ``preview``, one with a preview handle."""
        ...
