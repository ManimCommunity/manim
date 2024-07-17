from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from manim.mobject.opengl.opengl_mobject import OpenGLMobject


@dataclass(frozen=True, slots=True, eq=False)
class Updater:
    _updater: Callable[..., object]
    dt: bool = False

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Updater):
            return self._updater == other._updater
        return self._updater == other

    def __hash__(self) -> int:
        return hash((self._updater, self.dt))

    def __call__(self, mob: OpenGLMobject, dt: float) -> None:
        if self.dt:
            self._updater(mob, dt)
        else:
            self._updater(mob)
