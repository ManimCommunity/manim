"""Animations related to rotation."""

from __future__ import annotations

__all__ = ["Rotating", "Rotate"]

from typing import TYPE_CHECKING

from manim.animation.animation import Animation
from manim.constants import ORIGIN, OUT, PI, TAU
from manim.utils.rate_functions import linear

if TYPE_CHECKING:
    from manim.mobject.opengl.opengl_mobject import OpenGLMobject
    from manim.typing import Point3D, RateFunc, Vector3D


class Rotating(Animation):
    def __init__(
        self,
        mobject: OpenGLMobject,
        angle: float = TAU,
        axis: Vector3D = OUT,
        about_point: Point3D | None = None,
        about_edge: Vector3D | None = None,
        rate_func: RateFunc = linear,
        suspend_mobject_updating: bool = False,
        **kwargs,
    ):
        super().__init__(
            mobject,
            rate_func=rate_func,
            suspend_mobject_updating=suspend_mobject_updating,
            **kwargs,
        )
        self.angle = angle
        self.axis = axis
        self.about_point = about_point
        self.about_edge = about_edge

    def interpolate(self, alpha: float) -> None:
        pairs = zip(
            self.mobject.family_members_with_points(),
            self.starting_mobject.family_members_with_points(),
        )
        for sm1, sm2 in pairs:
            sm1.points[:] = sm2.points

        self.mobject.rotate(
            self.rate_func(alpha) * self.angle,
            axis=self.axis,
            about_point=self.about_point,
            about_edge=self.about_edge,
        )


class Rotate(Rotating):
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
        mobject: OpenGLMobject,
        angle: float = PI,
        axis: Vector3D = OUT,
        run_time: float = 1,
        about_edge: Vector3D = ORIGIN,
        **kwargs,
    ):
        super().__init__(
            mobject,
            angle,
            axis,
            run_time=run_time,
            about_edge=about_edge,
            introducer=True,
            **kwargs,
        )
