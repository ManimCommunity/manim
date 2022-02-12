"""Animations related to rotation."""

from __future__ import annotations

__all__ = ["Rotating", "Rotate"]

import typing
from typing import Callable, Sequence

import numpy as np

from ..animation.animation import Animation
from ..animation.transform import Transform
from ..constants import OUT, PI, TAU
from ..utils.rate_functions import linear

if typing.TYPE_CHECKING:
    from ..mobject.mobject import Mobject


class Rotating(Animation):
    def __init__(
        self,
        mobject: Mobject,
        axis: np.ndarray = OUT,
        radians: np.ndarray = TAU,
        about_point: np.ndarray | None = None,
        about_edge: np.ndarray | None = None,
        run_time: float = 5,
        rate_func: Callable[[float], float] = linear,
        **kwargs,
    ) -> None:
        self.axis = axis
        self.radians = radians
        self.about_point = about_point
        self.about_edge = about_edge
        super().__init__(mobject, run_time=run_time, rate_func=rate_func, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.become(self.starting_mobject)
        self.mobject.rotate(
            self.rate_func(alpha) * self.radians,
            axis=self.axis,
            about_point=self.about_point,
            about_edge=self.about_edge,
        )


class Rotate(Transform):
    def __init__(
        self,
        mobject: Mobject,
        angle: np.ndarray = PI,
        axis: np.ndarray = OUT,
        about_point: Sequence[float] | None = None,
        about_edge: Sequence[float] | None = None,
        **kwargs,
    ) -> None:
        if "path_arc" not in kwargs:
            kwargs["path_arc"] = angle
        if "path_arc_axis" not in kwargs:
            kwargs["path_arc_axis"] = axis
        self.angle = angle
        self.axis = axis
        self.about_edge = about_edge
        self.about_point = about_point
        if self.about_point is None:
            self.about_point = mobject.get_center()
        super().__init__(mobject, path_arc_centers=self.about_point, **kwargs)

    def create_target(self) -> Mobject:
        target = self.mobject.copy()
        target.rotate(
            self.angle,
            axis=self.axis,
            about_point=self.about_point,
            about_edge=self.about_edge,
        )
        return target
