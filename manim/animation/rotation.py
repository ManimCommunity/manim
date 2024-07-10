"""Animations related to rotation."""

from __future__ import annotations

__all__ = ["Rotating", "Rotate"]

from typing import TYPE_CHECKING, Callable

import numpy as np

from ..animation.animation import Animation
from ..constants import ORIGIN, OUT, PI, TAU
from ..utils.rate_functions import linear

if TYPE_CHECKING:
    from ..mobject.opengl.opengl_mobject import OpenGLMobject


class Rotating(Animation):
    def __init__(
        self,
        mobject: OpenGLMobject,
        angle: float = TAU,
        axis: np.ndarray = OUT,
        about_point: np.ndarray | None = None,
        about_edge: np.ndarray | None = None,
        run_time: float = 5.0,
        rate_func: Callable[[float], float] = linear,
        suspend_mobject_updating: bool = False,
        **kwargs,
    ):
        self.angle = angle
        self.axis = axis
        self.about_point = about_point
        self.about_edge = about_edge
        super().__init__(
            mobject,
            run_time=run_time,
            rate_func=rate_func,
            suspend_mobject_updating=suspend_mobject_updating,
            **kwargs,
        )

    def interpolate_mobject(self, alpha: float) -> None:
        pairs = zip(
            self.mobject.family_members_with_points(),
            self.starting_mobject.family_members_with_points(),
        )
        self.mobject.rotate(
            self.rate_func(alpha) * self.angle,
            axis=self.axis,
            about_point=self.about_point,
            about_edge=self.about_edge,
        )


class Rotate(Rotating):
    def __init__(
        self,
        mobject: OpenGLMobject,
        angle: float = PI,
        axis: np.ndarray = OUT,
        run_time: float = 1,
        about_edge: np.ndarray = ORIGIN,
        **kwargs,
    ):
        super().__init__(
            mobject, angle, axis, run_time=run_time, about_edge=about_edge, **kwargs
        )
