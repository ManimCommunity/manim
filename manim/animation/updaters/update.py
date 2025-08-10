"""Animations that update mobjects."""

from __future__ import annotations

__all__ = ["UpdateFromFunc", "UpdateFromAlphaFunc", "MaintainPositionRelativeTo"]


import operator as op
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from manim.animation.animation import Animation

if TYPE_CHECKING:
    from manim.mobject.mobject import Mobject


class UpdateFromFunc(Animation):
    """
    update_function of the form func(mobject), presumably
    to be used when the state of one mobject is dependent
    on another simultaneously animated mobject
    """

    def __init__(
        self,
        mobject: Mobject,
        update_function: Callable[[Mobject], Any],
        suspend_mobject_updating: bool = False,
        **kwargs: Any,
    ) -> None:
        self.update_function = update_function
        super().__init__(
            mobject, suspend_mobject_updating=suspend_mobject_updating, **kwargs
        )

    def interpolate_mobject(self, alpha: float) -> None:
        self.update_function(self.mobject)  # type: ignore[arg-type]


class UpdateFromAlphaFunc(UpdateFromFunc):
    def interpolate_mobject(self, alpha: float) -> None:
        self.update_function(self.mobject, self.rate_func(alpha))  # type: ignore[call-arg, arg-type]


class MaintainPositionRelativeTo(Animation):
    def __init__(
        self, mobject: Mobject, tracked_mobject: Mobject, **kwargs: Any
    ) -> None:
        self.tracked_mobject = tracked_mobject
        self.diff = op.sub(
            mobject.get_center(),
            tracked_mobject.get_center(),
        )
        super().__init__(mobject, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        target = self.tracked_mobject.get_center()
        location = self.mobject.get_center()
        self.mobject.shift(target - location + self.diff)
