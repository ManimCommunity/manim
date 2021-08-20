"""Animations related to rotation."""

__all__ = ["Rotating", "Rotate"]

import typing
from typing import Callable, Optional, Sequence

import numpy as np

from ..animation.animation import Animation
from ..animation.transform import Transform
from ..constants import OUT, PI, TAU
from ..utils.rate_functions import linear
from ..utils.space_ops import rotation_matrix_transpose

if typing.TYPE_CHECKING:
    from ..mobject.mobject import Mobject


class Rotating(Animation):
    def __init__(
        self,
        mobject: "Mobject",
        axis: np.ndarray = OUT,
        radians: np.ndarray = TAU,
        about_point: Optional[np.ndarray] = None,
        about_edge: Optional[np.ndarray] = None,
        run_time: float = 5,
        rate_func: Callable[[float], float] = linear,
        **kwargs
    ) -> None:
        self.axis = axis
        self.radians = radians
        self.about_point = about_point
        self.about_edge = about_edge
        super().__init__(mobject, run_time=run_time, rate_func=rate_func, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.become(self.starting_mobject)
        self.mobject.rotate(
            alpha * self.radians,
            axis=self.axis,
            about_point=self.about_point,
            about_edge=self.about_edge,
        )


class Rotate(Transform):
    def __init__(
        self,
        mobject: "Mobject",
        angle: np.ndarray = PI,
        axis: np.ndarray = OUT,
        about_point: Optional[Sequence[float]] = None,
        about_edge: Optional[Sequence[float]] = None,
        **kwargs
    ) -> None:
        if "path_arc" not in kwargs:
            kwargs["path_arc"] = angle
        if "path_arc_axis" not in kwargs:
            kwargs["path_arc_axis"] = axis
        self.angle = angle
        self.axis = axis
        self.about_edge = about_edge
        if about_point is None:
            self.about_point = mobject.get_center()
        self.path_func = self.get_rotation_arc_func(self.angle, self.about_point)
        super().__init__(mobject, path_func=self.path_func, **kwargs)

    def get_rotation_arc_func(self, angle, center):
        def func(start_points, end_points, alpha):
            rot_matrix_T = rotation_matrix_transpose(alpha * angle, OUT)
            return center + np.dot(start_points - center, rot_matrix_T)

        return func

    def create_target(self) -> "Mobject":
        target = self.mobject.copy()
        target.rotate(
            self.angle,
            axis=self.axis,
            about_point=self.about_point,
            about_edge=self.about_edge,
        )
        return target
