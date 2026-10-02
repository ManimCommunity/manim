"""Animations related to rotation."""

from __future__ import annotations

__all__ = ["Rotating", "Rotate"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..animation.animation import Animation
from ..animation.transform import Transform
from ..constants import OUT, PI, TAU
from ..utils.rate_functions import linear

if TYPE_CHECKING:
    from ..mobject.mobject import Mobject
    from ..mobject.opengl.opengl_mobject import OpenGLMobject
    from ..typing import Point3DLike, Vector3DLike


class Rotating(Animation):
    """Animation that rotates a Mobject.

    Parameters
    ----------
    mobject
        The mobject to be rotated.
    angle
        The rotation angle in radians. Predefined constants such as ``DEGREES``
        can also be used to specify the angle in degrees.
    axis
        The rotation axis as a numpy vector.
    about_point
        The rotation center.
    about_edge
        If ``about_point`` is ``None``, this argument specifies
        the direction of the bounding box point to be taken as
        the rotation center.
    run_time
        The duration of the animation in seconds.
    rate_func
        The function defining the animation progress based on the relative
        runtime (see :mod:`~.rate_functions`) .
    **kwargs
        Additional keyword arguments passed to :class:`~.Animation`.

    Examples
    --------
    .. manim:: RotatingDemo

        class RotatingDemo(Scene):
            def construct(self):
                circle = Circle(radius=1, color=BLUE)
                line = Line(start=ORIGIN, end=RIGHT)
                arrow = Arrow(start=ORIGIN, end=RIGHT, buff=0, color=GOLD)
                vg = VGroup(circle,line,arrow)
                self.add(vg)
                anim_kw = {"about_point": arrow.get_start(), "run_time": 1}
                self.play(Rotating(arrow, 180*DEGREES, **anim_kw))
                self.play(Rotating(arrow, PI, **anim_kw))
                self.play(Rotating(vg, PI, about_point=RIGHT))
                self.play(Rotating(vg, PI, axis=UP, about_point=ORIGIN))
                self.play(Rotating(vg, PI, axis=RIGHT, about_edge=UP))
                self.play(vg.animate.move_to(ORIGIN))

    .. manim:: RotatingDifferentAxis

        class RotatingDifferentAxis(ThreeDScene):
            def construct(self):
                axes = ThreeDAxes()
                cube = Cube()
                arrow2d = Arrow(start=[0, -1.2, 1], end=[0, 1.2, 1], color=YELLOW_E)
                cube_group = VGroup(cube,arrow2d)
                self.set_camera_orientation(gamma=0, phi=40*DEGREES, theta=40*DEGREES)
                self.add(axes, cube_group)
                play_kw = {"run_time": 1.5}
                self.play(Rotating(cube_group, PI), **play_kw)
                self.play(Rotating(cube_group, PI, axis=UP), **play_kw)
                self.play(Rotating(cube_group, 180*DEGREES, axis=RIGHT), **play_kw)
                self.wait(0.5)

    See also
    --------
    :class:`~.Rotate`, :meth:`~.Mobject.rotate`

    """

    def __init__(
        self,
        mobject: Mobject,
        angle: float = TAU,
        axis: Vector3DLike = OUT,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        run_time: float = 5,
        rate_func: Callable[[float], float] = linear,
        **kwargs: Any,
    ) -> None:
        self.angle = angle
        self.axis = axis
        self.about_point = about_point
        self.about_edge = about_edge
        super().__init__(mobject, run_time=run_time, rate_func=rate_func, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.become(self.starting_mobject)
        self.mobject.rotate(
            self.rate_func(alpha) * self.angle,
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

    See also
    --------
    :class:`~.Rotating`, :meth:`~.Mobject.rotate`

    """

    def __init__(
        self,
        mobject: Mobject,
        angle: float = PI,
        axis: Vector3DLike = OUT,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
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

    def create_target(self) -> Mobject | OpenGLMobject:
        target = self.mobject.copy()
        target.rotate(
            self.angle,
            axis=self.axis,
            about_point=self.about_point,
            about_edge=self.about_edge,
        )
        return target
