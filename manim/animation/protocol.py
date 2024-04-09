from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from .animation import Animation
    from .scene_buffer import SceneBuffer


__all__ = ("AnimationProtocol",)


class AnimationProtocol(Protocol):
    buffer: SceneBuffer

    def begin(self) -> None:
        ...

    def finish(self) -> None:
        ...

    def get_all_animations(self) -> Sequence[Animation]:
        ...

    def update_mobjects(self, dt: float) -> None:
        ...

    def interpolate(self, alpha: float) -> None:
        ...

    def get_run_time(self) -> float:
        ...
