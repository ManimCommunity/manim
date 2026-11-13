"""Renderer capabilities, lifecycle state, and drawing methods used by Manager."""

from __future__ import annotations

__all__ = ["Renderer", "RendererCapabilities", "RendererState"]

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject
    from manim.scene.scene import Scene
    from manim.typing import RGBAPixelArray


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

    def _prepare_animation(self, scene: Scene) -> None: ...

    def render(
        self, scene: Scene, frame_offset: float, moving_mobjects: list[Mobject], /
    ) -> None: ...

    def get_frame(self) -> RGBAPixelArray: ...

    def _present_frame(self, scene: Scene, frame_offset: float) -> None: ...

    def _present_frozen_frame(self, scene: Scene, duration: float) -> None: ...
