"""Animations related to rotation."""

from __future__ import annotations

__all__ = ["Rotating", "Rotate"]

from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable

import numpy as np

from ..animation.animation import Animation
from ..animation.transform import Transform
from ..constants import OUT, PI, TAU
from ..utils.rate_functions import linear

if TYPE_CHECKING:
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
    """Animation that rotates a Mobject.

    Parameters
    ----------
    mobject
        The mobject to be rotated.
    angle
        The rotation angle.
    axis
        The rotation axis as a numpy vector.
    about_point
        The rotation center.
    about_edge
        If ``about_point`` is ``None``, this argument specifies
        the direction of the bounding box point to be taken as
        the rotation center.

    Examples
    --------
    .. manim:: UsingRotate

        class UsingRotate(Scene):
            def construct(self):
                self.play(
                    Rotate(
                        Square(side_length=0.5).shift(UP * 2),
                        angle=2*PI,
                        about_point=ORIGIN,
                        rate_func=linear,
                    ),
                    Rotate(Square(side_length=0.5), angle=2*PI, rate_func=linear),
                    )

    """

    def __init__(
        self,
        mobject: Mobject,
        angle: float = PI,
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
