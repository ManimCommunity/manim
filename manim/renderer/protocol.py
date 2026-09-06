"""Shared renderer features and the narrow animation drawing boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject
    from manim.scene.scene import Scene
    from manim.typing import RGBAPixelArray

__all__ = ["RendererCapabilities"]


@dataclass(frozen=True, slots=True)
class RendererCapabilities:
    """Optional session features implemented by a renderer."""

    live_preview: bool = False


class _AnimationRenderer(Protocol):
    """Drawing/presentation operations, never compilation, selection or clock ticks.

    Kept private during the render-flow migration. The native backend owns wall
    pacing; Manager owns when these operations are requested and any output delivery.
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
