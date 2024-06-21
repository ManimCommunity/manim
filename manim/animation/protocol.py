from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .scene_buffer import SceneBuffer


__all__ = ("AnimationProtocol",)


class AnimationProtocol(Protocol):
    buffer: SceneBuffer
    apply_buffer: bool

    def begin(self) -> None: ...

    def finish(self) -> None: ...

    def update_mobjects(self, dt: float) -> None: ...

    def interpolate(self, alpha: float) -> None: ...

    def get_run_time(self) -> float: ...
