"""Animations that introduce mobjects to scene by growing them from points.

.. manim:: Growing

    class Growing(Scene):
        def construct(self):
            square = Square()
            circle = Circle()
            triangle = Triangle()
            arrow = Arrow(LEFT, RIGHT)
            star = Star()

            VGroup(square, circle, triangle).set_x(0).arrange(buff=1.5).set_y(2)
            VGroup(arrow, star).move_to(DOWN).set_x(0).arrange(buff=1.5).set_y(-2)

            self.play(GrowFromPoint(square, ORIGIN))
            self.play(GrowFromCenter(circle))
            self.play(GrowFromEdge(triangle, DOWN))
            self.play(GrowArrow(arrow))
            self.play(SpinInFromNothing(star))

"""

from __future__ import annotations

__all__ = [
    "GrowFromPoint",
    "GrowFromCenter",
    "GrowFromEdge",
    "GrowArrow",
    "SpinInFromNothing",
]

import typing

import numpy as np

from ..animation.transform import Transform
from ..constants import PI
from ..utils.paths import spiral_path

if typing.TYPE_CHECKING:
    from manim.mobject.geometry.line import Arrow

    from ..mobject.mobject import Mobject


class GrowFromPoint(Transform):
    """Introduce an :class:`~.Mobject` by growing it from a point.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    point
        The point from which the mobject grows.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: GrowFromPointExample

        class GrowFromPointExample(Scene):
            def construct(self):
                dot = Dot(3 * UR, color=GREEN)
                squares = [Square() for _ in range(4)]
                VGroup(*squares).set_x(0).arrange(buff=1)
                self.add(dot)
                self.play(GrowFromPoint(squares[0], ORIGIN))
                self.play(GrowFromPoint(squares[1], [-2, 2, 0]))
                self.play(GrowFromPoint(squares[2], [3, -2, 0], RED))
                self.play(GrowFromPoint(squares[3], dot, dot.get_color()))

    """

    def __init__(
        self, mobject: Mobject, point: np.ndarray, point_color: str = None, **kwargs
    ) -> None:
        self.point = point
        self.point_color = point_color
        super().__init__(mobject, introducer=True, **kwargs)

    def create_target(self) -> Mobject:
        return self.mobject

    def create_starting_mobject(self) -> Mobject:
        start = super().create_starting_mobject()
        start.scale(0)
        start.move_to(self.point)
        if self.point_color:
            start.set_color(self.point_color)
        return start


class GrowFromCenter(GrowFromPoint):
    """Introduce an :class:`~.Mobject` by growing it from its center.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: GrowFromCenterExample

        class GrowFromCenterExample(Scene):
            def construct(self):
                squares = [Square() for _ in range(2)]
                VGroup(*squares).set_x(0).arrange(buff=2)
                self.play(GrowFromCenter(squares[0]))
                self.play(GrowFromCenter(squares[1], point_color=RED))

    """

    def __init__(self, mobject: Mobject, point_color: str = None, **kwargs) -> None:
        point = mobject.get_center()
        super().__init__(mobject, point, point_color=point_color, **kwargs)


class GrowFromEdge(GrowFromPoint):
    """Introduce an :class:`~.Mobject` by growing it from one of its bounding box edges.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    edge
        The direction to seek bounding box edge of mobject.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: GrowFromEdgeExample

        class GrowFromEdgeExample(Scene):
            def construct(self):
                squares = [Square() for _ in range(4)]
                VGroup(*squares).set_x(0).arrange(buff=1)
                self.play(GrowFromEdge(squares[0], DOWN))
                self.play(GrowFromEdge(squares[1], RIGHT))
                self.play(GrowFromEdge(squares[2], UR))
                self.play(GrowFromEdge(squares[3], UP, point_color=RED))


    """

    def __init__(
        self, mobject: Mobject, edge: np.ndarray, point_color: str = None, **kwargs
    ) -> None:
        point = mobject.get_critical_point(edge)
        super().__init__(mobject, point, point_color=point_color, **kwargs)


class GrowArrow(GrowFromPoint):
    """Introduce an :class:`~.Arrow` by growing it from its start toward its tip.

    Parameters
    ----------
    arrow
        The arrow to be introduced.
    point_color
        Initial color of the arrow before growing to its full size. Leave empty to match arrow's color.

    Examples
    --------

    .. manim :: GrowArrowExample

        class GrowArrowExample(Scene):
            def construct(self):
                arrows = [Arrow(2 * LEFT, 2 * RIGHT), Arrow(2 * DR, 2 * UL)]
                VGroup(*arrows).set_x(0).arrange(buff=2)
                self.play(GrowArrow(arrows[0]))
                self.play(GrowArrow(arrows[1], point_color=RED))

    """

    def __init__(self, arrow: Arrow, point_color: str = None, **kwargs) -> None:
        point = arrow.get_start()
        super().__init__(arrow, point, point_color=point_color, **kwargs)

    def create_starting_mobject(self) -> Mobject:
        start_arrow = self.mobject.copy()
        start_arrow.scale(0, scale_tips=True, about_point=self.point)
        if self.point_color:
            start_arrow.set_color(self.point_color)
        return start_arrow


class SpinInFromNothing(GrowFromCenter):
    """Introduce an :class:`~.Mobject` spinning and growing it from its center.

    Parameters
    ----------
    mobject
        The mobjects to be introduced.
    angle
        The amount of spinning before mobject reaches its full size. E.g. 2*PI means
        that the object will do one full spin before being fully introduced.
    point_color
        Initial color of the mobject before growing to its full size. Leave empty to match mobject's color.

    Examples
    --------

    .. manim :: SpinInFromNothingExample

        class SpinInFromNothingExample(Scene):
            def construct(self):
                squares = [Square() for _ in range(3)]
                VGroup(*squares).set_x(0).arrange(buff=2)
                self.play(SpinInFromNothing(squares[0]))
                self.play(SpinInFromNothing(squares[1], angle=2 * PI))
                self.play(SpinInFromNothing(squares[2], point_color=RED))

    """

    def __init__(
        self, mobject: Mobject, angle: float = PI / 2, point_color: str = None, **kwargs
    ) -> None:
        self.angle = angle
        super().__init__(
            mobject, path_func=spiral_path(angle), point_color=point_color, **kwargs
        )
