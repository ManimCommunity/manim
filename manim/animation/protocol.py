from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from manim.typing import RateFunc

    from .scene_buffer import SceneBuffer


__all__ = ("AnimationProtocol",)


class AnimationProtocol(Protocol):
    buffer: SceneBuffer
    apply_buffer: bool

    def begin(self) -> object: ...

    def finish(self) -> object: ...

    def update_mobjects(self, dt: float) -> object: ...

    def interpolate(self, alpha: float) -> object: ...

    def get_run_time(self) -> float: ...

    def update_rate_info(
        self,
        run_time: float | None,
        rate_func: RateFunc | None,
        lag_ratio: float | None,
    ) -> object: ...
