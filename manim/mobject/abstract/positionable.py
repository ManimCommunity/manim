from __future__ import annotations

import operator
import sys
from collections.abc import Callable, Iterable
from functools import reduce
from typing import Any, Literal, Self

import numpy as np

from manim._config import config
from manim.constants import (
    DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
    DL,
    DOWN,
    IN,
    LEFT,
    MED_SMALL_BUFF,
    ORIGIN,
    OUT,
    RIGHT,
    TAU,
    UP,
    UR,
)
from manim.typing import (
    MatrixMN,
    Point3D,
    Point3D_Array,
    Point3DLike,
    Point3DLike_Array,
    Vector3D,
    Vector3DLike,
)
from manim.utils.space_ops import rotation_matrix

__all__ = ["Positionable"]


class Positionable:
    """A positionable object.

    Attributes
    ----------
    points
        The points making up the body of the object.
    """

    # =============
    # region POINTS
    # =============

    points: Point3D_Array = np.zeros((0, 3))

    def get_all_points(self) -> Point3D_Array:
        """Returns all points of the object.

        Returns
        -------
        Point3D_Array
            The points.
        """
        all_points = [mob.points for mob in self.get_family() if mob.has_points()]
        if len(all_points) == 0:
            return np.zeros((0, 3))
        elif len(all_points) == 1:
            return all_points[0]
        return np.concatenate(all_points)

    def set_points(self, points: Point3DLike_Array) -> Self:
        """Replaces the object's points with ``points``.

        Does not affect family members.

        Parameters
        ----------
        points
            The points.

        Returns
        -------
        Self
            The object itself.
        """
        # TODO: Do we need to create a copy?
        self.points = np.asarray(points, dtype=float, copy=True)
        if self.points.size == 0:
            self.points = self.points.reshape(0, 3)
        return self

    def match_points(
        self,
        other: Positionable,
        *,
        strict: bool = False,
    ) -> Self:
        """Replaces the points of the object so that they are identical to the other object.

        The points of the family members are matched in order.

        Parameters
        ----------
        other
            The other object.
        strict
            Whether to through an error when the familys are not of the same size.
            Defaults to ``False``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: MatchPointsExample

            class MatchPointsExample(Scene):
                def construct(self):
                    circ = Circle(fill_color=RED, fill_opacity=0.8)
                    square = Square(fill_color=BLUE, fill_opacity=0.2)
                    self.add(circ)
                    self.play(circ.animate.match_points(square))
        """
        for sm1, sm2 in zip(self.get_family(), other.get_family(), strict=strict):
            sm1.points = sm2.points.copy()
        return self

    def reset_points(self, **kwargs: Any) -> Self:
        """Resets the points of the object.

        Does not affect family members.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_points(np.zeros((0, 3)), **kwargs)

    def reverse_points(self, **kwargs: Any) -> Self:
        """Reverses the points of the object.

        Returns
        -------
        Self
            The object itself.


        Example
        -------
        .. manim:: ReversePointsExample

            class ReversePointsExample(Scene):
                def construct(self) -> None:
                    star_1 = Star().set_color(RED)
                    star_2 = Star().set_color(GREEN).reverse_points()
                    VGroup(star_1, star_2).arrange()
                    self.play(Create(star_1), Create(star_2), run_time=3)
        """

        def apply(mob: Positionable) -> None:
            mob.points[:] = mob.points[::-1]

        return self.apply_to_family(apply, **kwargs)

    def repeat_points(self, count: int, **kwargs: Any) -> Self:
        """Repeats the points of the object.

        Can make transition animations nicer.

        Parameters
        ----------
        count
            The repeat count.

        Returns
        -------
        Self
            The object itself.
        """

        def apply(mob: Positionable) -> None:
            mob.points = np.tile(mob.points, (count, 1))

        return self.apply_to_family(apply, **kwargs)

    def get_num_points(self) -> int:
        """Returns the number of points of the object.

        Does not take family members into account.

        Returns
        -------
        int
            The number of points.
        """
        return len(self.points)

    def has_points(self) -> bool:
        """Whether the object has points.

        Does not take family members into account.

        Returns
        -------
        bool
            Has points.
        """
        return len(self.points) != 0

    # =========
    # endregion
    # =========

    # =========================
    # region APPLYING FUNCTIONS
    # =========================

    def get_family(self) -> list[Positionable]:
        """Returns the family of the object.

        Each member is only included once.

        Returns
        -------
        list[Positionable]
            The family members.

        Example
        -------
        ::

            >>> from manim import Square, Rectangle, VGroup, Group, Mobject, VMobject
            >>> s, r, m, v = Square(), Rectangle(), Mobject(), VMobject()
            >>> vg = VGroup(s, r)
            >>> gr = Group(vg, m, v)
            >>> gr.get_family()
            [Group, VGroup(Square, Rectangle), Square, Rectangle, Mobject, VMobject]
        """
        return [self]

    def apply_to_family(
        self,
        function: Callable[[Positionable], Any],
        *,
        should_skip: Callable[[Positionable], bool] = lambda mob: not mob.has_points(),
        **kwargs: Any,
    ) -> Self:
        """Applies a function to the object and its family members.

        Parameters
        ----------
        function
            The function to be applied.
        should_skip
            A predicate function which returns ``True`` if a family member should be skipped.
            Defaults to skipping family members with no points

        Returns
        -------
        Self
            The object itself.


        Example
        -------
        .. manim:: ApplyToFamilyExample

            class ApplyToFamilyExample(Scene):
                def construct(self) -> None:
                    group = VGroup(Circle() for _ in range(4)).arrange_in_grid()
                    def function(mob: VMobject) -> None:
                        mob.move_to(ORIGIN)
                    self.play(group.animate.apply_to_family(function))

        """
        for mob in self.get_family():
            if not should_skip(mob):
                function(mob)

        return self

    def apply_points_function(
        self,
        function: Callable[[Point3D_Array], Point3D_Array],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a function to the points of the object.

        Parameters
        ----------
        function
            The function to be applied.
        about_point
            About which point to apply the function.
            Defaults to ``None``
        about_edge
            About which edge to apply the function.
            Defaults to ``None``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ApplyPointsFunctionExample

            class ApplyPointsFunctionExample(Scene):
                def construct(self) -> None:
                    circle = Circle()

                    def function(points: Point3D_Array) -> Point3D_Array:
                        points[:, 0] *= 3
                        points[:, 1] = -abs(points[:, 1])
                        return points

                    self.play(circle.animate.apply_points_function(function))
        """
        about_point = self._get_about_point(about_point, about_edge)

        def apply(mob: Positionable) -> None:
            mob.points -= about_point
            mob.points = function(mob.points)
            mob.points += about_point

        return self.apply_to_family(apply, **kwargs)

    # TODO: Rename to `apply_point_function`?
    def apply_function(
        self,
        function: Callable[[Point3D], Point3D],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a function to every point of the object.

        Parameters
        ----------
        function
            The function to be applied.
        about_point
            About which point to apply the function.
            Defaults to ``None``
        about_edge
            About which edge to apply the function.
            Defaults to ``None``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ApplyPointFunctionExample

            class ApplyPointFunctionExample(Scene):
                def construct(self) -> None:
                    circle = Circle()

                    def function(point: Point3D) -> Point3D:
                        return np.array([point[0], np.cos(point[1]) - 1, point[2]])

                    self.play(circle.animate.apply_function(function))
        """
        return self.apply_points_function(
            lambda points: np.apply_along_axis(function, 1, points),
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def apply_complex_function(
        self,
        function: Callable[[complex], complex],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a complex function to every point of the object.

        Parameters
        ----------
        function
            The function to be applied.
        about_point
            About which point to apply the function.
            Defaults to ``None``
        about_edge
            About which edge to apply the function.
            Defaults to ``None``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ApplyComplexFunctionExample

            class ApplyComplexFunctionExample(Scene):
                def construct(self):
                    circ = Circle().scale(1.5)
                    circ_ref = circ.copy()
                    circ.apply_complex_function(
                        lambda x: np.exp(x*1j)
                    )
                    t = ValueTracker(0)
                    circ.add_updater(
                        lambda x: x.become(circ_ref.copy().apply_complex_function(
                            lambda x: np.exp(x+t.get_value()*1j)
                        )).set_color(BLUE)
                    )
                    self.add(circ_ref)
                    self.play(TransformFromCopy(circ_ref, circ))
                    self.play(t.animate.set_value(TAU), run_time=3)
        """

        def R3_func(point: Point3D) -> Point3D:
            x, y, z = point
            xy_complex = function(complex(x, y))
            return np.array([xy_complex.real, xy_complex.imag, z])

        return self.apply_function(
            R3_func,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # =========
    # endregion
    # =========

    # ======================
    # region TRANSFORMATIONS
    # ======================

    def translate(
        self,
        vector: Vector3DLike,
        **kwargs: Any,
    ) -> Self:
        """Translates the object by a vector.

        Parameters
        ----------
        vector
            The translation vector.

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: TranslateExample

            class TranslateExample(Scene):
                def construct(self) -> None:
                    circle = Circle()
                    circle.translate(LEFT + UP)
                    self.play(circle.animate.translate(2 * RIGHT))
                    self.play(circle.animate.translate(2 * DOWN))
                    self.play(circle.animate.translate(2 * LEFT))
                    self.play(circle.animate.translate(2 * UP))
        """

        def apply(mob: Positionable) -> None:
            mob.points += vector

        return self.apply_to_family(apply, **kwargs)

    def scale(
        self,
        factor: float | Vector3DLike,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        r"""Scales the object uniformly along all dimensions.

        Parameters
        ----------
        factor
            The scaling factor.

            - For :math:`0 < |\alpha| < 1` the object shrinks.
            - For :math:`|\alpha| > 1` the object grows.
            - For :math:`\alpha < 0` the object flips.
        about_point
            About which point to scale.
            Defaults to ``None``
        about_edge
            About which edge to scale.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ScaleExample
            :save_last_frame:

            class ScaleExample(Scene):
                def construct(self):
                    f1 = Text("F")
                    f2 = Text("F").scale(2)
                    f3 = Text("F").scale(0.5)
                    f4 = Text("F").scale(-1)

                    vgroup = VGroup(f1, f2, f3, f4).arrange(6 * RIGHT)
                    self.add(vgroup)
        """
        factor = np.asarray(factor)

        def apply(points: Point3D_Array) -> Point3D_Array:
            return factor * points

        return self.apply_points_function(
            apply,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch(
        self,
        factor: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        r"""Stretches the object along one dimension
        while leaving the other dimensions untouched.

        Parameters
        ----------
        factor
            The stretching factor.

            - For :math:`0 < |\alpha| < 1` the object shrinks.
            - For :math:`|\alpha| > 1` the object grows.
            - For :math:`\alpha < 0` the object flips.
        dim
            The dimension to stretch along.
        about_point
            About which point to stretch.
            Defaults to ``None``
        about_edge
            About which edge to stretch.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: StretchExample

            class StretchExample(Scene):
                def construct(self) -> None:
                    circle = Circle()
                    self.play(circle.animate.stretch(3, dim=0))
                    self.play(circle.animate.stretch(0.5, dim=1))
        """

        def apply(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        return self.apply_points_function(
            apply,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Rotates the object by an angle along an axis.

        Parameters
        ----------
        angle
            The rotation angle in radians.
            The predefined constant ``DEGREES`` can be used to specify the angles in degrees.
        axis
            The axis to rotate around.
            Defaults to ``OUT``, meaning the Z axis (i.e. the XY plane)
        about_point
            About which point to rotate.
            Defaults to ``None``
        about_edge
            About which edge to rotate.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        .. note::
            To animate a rotation, use :class:`~.Rotating` or :class:`~.Rotate`
            instead of ``.animate.rotate(...)``.
            The ``.animate.rotate(...)`` syntax only applies a transformation
            from the initial state to the final rotated state
            (interpolation between the two states), without showing proper rotational motion
            based on the angle (from 0 to the given angle).

        Example
        -------
        .. manim:: RotateExample
            :save_last_frame:

            class RotateExample(Scene):
                def construct(self):
                    circle = Circle(radius=1, color=BLUE)
                    line = Line(start=ORIGIN, end=RIGHT)
                    arrow1 = Arrow(start=ORIGIN, end=RIGHT, buff=0, color=GOLD)
                    group1 = VGroup(circle, line, arrow1)

                    group2 = group1.copy()
                    arrow2 = group2[2]
                    arrow2.rotate(angle=PI / 4, about_point=arrow2.get_start())

                    group3 = group1.copy()
                    arrow3 = group3[2]
                    arrow3.rotate(angle=120 * DEGREES, about_point=arrow3.get_start())

                    self.add(VGroup(group1, group2, group3).arrange(RIGHT, buff=1))
        """
        return self.apply_matrix(
            rotation_matrix(angle, axis),
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def apply_matrix(
        self,
        matrix: MatrixMN,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a transformation matrix to the points of the object.

        Parameters
        ----------
        matrix
            The transformation matrix.
        about_point
            About which point to apply the matrix.
            Defaults to ``None``
        about_edge
            About which edge to apply the matrix.
            Defaults to ``None``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ApplyMatrixExample

            class ApplyMatrixExample(Scene):
                def construct(self) -> None:
                    scale_x = 2
                    scale_y = 3
                    shear_x = 120 * DEGREES
                    matrix = np.array(
                        [
                            [scale_x, shear_x, 0],
                            [0, scale_y, 0],
                            [0, 0, 1],
                        ],
                    )
                    self.play(Star().animate.apply_matrix(matrix))
        """
        if matrix.shape == (3, 3):
            full_matrix = matrix
        else:
            full_matrix = np.identity(3)
            full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

        def apply(points: Point3D_Array) -> Point3D_Array:
            return points.dot(full_matrix.T)

        return self.apply_points_function(
            apply,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def _get_about_point(
        self,
        about_point: Point3DLike | None,
        about_edge: Vector3DLike | None,
    ) -> Point3DLike:
        if about_point is not None:
            # TODO: Do we really need this?
            # Make a copy to prevent mutation of the original array if about_point is a view
            return np.asarray(about_point, copy=True)
        elif about_edge is not None:
            return self.get_anchor(about_edge)
        else:
            return ORIGIN.copy()

    # =========
    # endregion
    # =========

    # ===============
    # region POSITION
    # ===============

    def get_anchor(self, direction: Vector3DLike = ORIGIN) -> Point3D:
        """Returns the position of an anchor of the object.

        Parameters
        ----------
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Point3D
            The anchor position.

        .. note::
            Picture an axis-aligned bounding box which surrounds the object.
            The object's "anchor points" are the 27 points located at the corners, edge centers, face centers, and the center of this bounding box.
            Each anchor point's direction is defined as a vector whose dimensions are of value -1, 0 or 1.
            Manim offers predefined constants for many of these directions, including all on the XY plane:

            .. code-block::

                UL   |   UP   |    UR
                -----|--------|------
                LEFT | ORIGIN | RIGHT
                -----|--------|------
                DL   |  DOWN  |    DR

        Example
        -------
        .. manim:: GetAnchorExample
            :save_last_frame:

            class GetAnchorExample(Scene):
                def construct(self) -> None:
                    circle = Circle(radius=2)
                    self.add(circle)

                    anchors = [
                        ("UL", UL),         ("UP", UP),           ("UR", UR),
                        ("LEFT", LEFT), ("ORIGIN", ORIGIN), ("RIGHT", RIGHT),
                        ("DL", DL),       ("DOWN", DOWN),         ("DR", DR),
                    ]
                    for label, anchor in anchors:
                        dot = Dot(color=BLUE)
                        dot.move_to(circle.get_anchor(anchor))
                        dot.add(Text(label, font_size=24).next_to(dot, DOWN if anchor is ORIGIN else anchor, buff=0.1))
                        self.add(dot)
        """
        all_points = self.get_all_points()
        return np.array(
            [
                self._get_extremum(all_points[:, dim], key)
                for dim, key in enumerate(direction)
            ]
        )

    def set_anchor(
        self,
        position: Point3DLike,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the anchor is at ``position``.

        Parameters
        ----------
        position
            The anchor position.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: SetAnchorExample

            class SetAnchorExample(Scene):
                def construct(self) -> None:
                    point = (0, -1, 0)
                    anchors = [LEFT, RIGHT, UP, DOWN]

                    circles = VGroup(Circle() for _ in range(len(anchors)))
                    circles.arrange(RIGHT).to_edge(UP)

                    self.add(Dot(point), circles)
                    for anchor, circle in zip(anchors, circles):
                        self.play(circle.animate.set_anchor(point, anchor))
        """
        source = self.get_anchor(direction)
        vector = position - source
        return self.translate(vector, **kwargs)

    def get_center(self) -> Point3D:
        """Returns the center position of the object.

        Returns
        -------
        Point3D
            The center position.

        Example
        -------
        .. manim:: GetCenterExample

            class GetCenterExample(Scene):
                def construct(self) -> None:
                    circle = Circle().to_corner(UL)
                    arrow = Arrow()

                    def update(mob: Arrow) -> Arrow:
                        return mob.put_start_and_end_on(ORIGIN, circle.get_center())

                    arrow.add_updater(update)
                    self.add(circle, arrow)
                    self.play(circle.animate.to_corner(UR))
        """
        return self.get_anchor(ORIGIN)

    def set_center(
        self,
        position: Point3DLike,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that its center position is at ``position``.

        Parameters
        ----------
        position
            The center position.

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: SetCenterExample

            class SetCenterExample(Scene):
                def construct(self) -> None:
                    circle = Circle()
                    square = Square()

                    self.add(Dot((-2, 0, 0)), Dot((2, 0, 0)))
                    self.play(
                        circle.animate.set_center((-2, 0, 0)),
                        square.animate.set_center((2, 0, 0)),
                    )
        """
        return self.set_anchor(position, ORIGIN, **kwargs)

    def center(self, **kwargs: Any) -> Self:
        """Translates the object so that its center position is at ``ORIGIN``.

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: CenterExample

            class CenterExample(Scene):
                def construct(self) -> None:
                    circle_0 = Circle().to_corner(UL)
                    circle_1 = Circle().to_corner(UR)
                    circle_2 = Circle().to_corner(DL)
                    circle_3 = Circle().to_corner(DR)

                    self.play(
                        circle_0.animate.center(),
                        circle_1.animate.center(),
                        circle_2.animate.center(),
                        circle_3.animate.center(),
                    )
        """
        return self.set_center(ORIGIN, **kwargs)

    def get_top(self) -> Point3D:
        """Returns the top position of the object.

        Returns
        -------
        Point3D
            The top position.
        """
        return self.get_anchor(UP)

    def set_top(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that its top position is at ``position``.

        Parameters
        ----------
        position
            The top position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, UP, **kwargs)

    def get_bottom(self) -> Point3D:
        """Returns the bottom position of the object.

        Returns
        -------
        Point3D
            The bottom position.
        """
        return self.get_anchor(DOWN)

    def set_bottom(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that its bottom position is at ``position``.

        Parameters
        ----------
        position
            The bottom position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, DOWN, **kwargs)

    def get_right(self) -> Point3D:
        """Returns the right position of the object.

        Returns
        -------
        Point3D
            The right position.
        """
        return self.get_anchor(RIGHT)

    def set_right(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that its right position is at ``position``.

        Parameters
        ----------
        position
            The right position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, RIGHT, **kwargs)

    def get_left(self) -> Point3D:
        """Returns the left position of the object.

        Returns
        -------
        Point3D
            The left position.
        """
        return self.get_anchor(LEFT)

    def set_left(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that its left position is at ``position``.

        Parameters
        ----------
        position
            The left position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, LEFT, **kwargs)

    def get_zenith(self) -> Point3D:
        """Returns the zenith position of the object.

        Returns
        -------
        Point3D
            The zenith position.
        """
        return self.get_anchor(OUT)

    def set_zenith(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that its zenith position is at ``position``.

        Parameters
        ----------
        position
            The zenith position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, OUT, **kwargs)

    def get_nadir(self) -> Point3D:
        """Returns the nadir position of the object.

        Returns
        -------
        Point3D
            The nadir position.
        """
        return self.get_anchor(IN)

    def set_nadir(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that its nadir position is at ``position``.

        Parameters
        ----------
        position
            The nadir position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, IN, **kwargs)

    def get_coordinate(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the ``dim``th coordinate at one of the object's anchor points.

        Parameters
        ----------
        dim
            The dimension of the coordinate.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        float
            The coordinate value.
        """
        return self._get_extremum(self.get_all_points()[:, dim], direction[dim])

    def set_coordinate(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the ``dim``th coordinate at the specified
        anchor point is at ``value``.

        Parameters
        ----------
        value
            The coordinate value.
        dim
            The dimension of the coordinate.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: SetCoordinateExample

            class SetCoordinateExample(Scene):
                def construct(self) -> None:
                    circle = Circle()
                    self.play(circle.animate.set_coordinate(-3, dim=0))
                    self.play(circle.animate.set_coordinate(2, dim=1))
        """
        source = self.get_coordinate(dim, direction)
        vector = np.zeros(3)
        vector[dim] = value - source
        return self.translate(vector, **kwargs)

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the x coordinate at one of the object's anchor points.

        Parameters
        ----------
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        float
            The x coordinate.
        """
        return self.get_coordinate(0, direction)

    def set_x(
        self, value: float, direction: Vector3DLike = ORIGIN, **kwargs: Any
    ) -> Self:
        """Translates the object so that its x coordinate at the specified anchor point
        is at ``value``.

        Parameters
        ----------
        value
            The x value.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, 0, direction, **kwargs)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the y coordinate at one of the object's anchor points.

        Parameters
        ----------
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        float
            The y coordinate.
        """
        return self.get_coordinate(1, direction)

    def set_y(
        self, value: float, direction: Vector3DLike = ORIGIN, **kwargs: Any
    ) -> Self:
        """Translates the object so that its y coordinate at the specified anchor point
        is at ``value``.

        Parameters
        ----------
        value
            The y value.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, 1, direction, **kwargs)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the z coordinate at one of the object's anchor points.

        Parameters
        ----------
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        float
            The z coordinate.
        """
        return self.get_coordinate(2, direction)

    def set_z(
        self,
        value: float,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that its z coordinate at the specified anchor point
        is at ``value``.

        Parameters
        ----------
        value
            The z value.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, 2, direction, **kwargs)

    def align_on_border(
        self,
        direction: Vector3DLike,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Aligns the object on a border of the camera's frame.

        For example, ``obj.align_on_border(UP, buff=1)`` moves ``obj`` vertically
        so that its top edge is 1 unit from the top edge of the frame. ``obj``
        is not moved along the x or z axes.

        .. note ::

            This method currently only works for stationary cameras. To properly
            align an object to a :class:`~.MovingCamera`, see
            :meth:`~.Positionable.align_to`.

        Parameters
        ----------
        direction
            The direction of the border.
        buff
            The distance to the border.
            Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: AlignOnBorderExample

            class AlignOnBorderExample(Scene):
                def construct(self) -> None:
                    circle, triangle, square, star = Circle(), Triangle(), Square(), Star()
                    VGroup(circle, triangle, square, star).arrange()
                    self.add(circle, triangle, square, star)

                    self.play(circle.animate.align_on_border(LEFT))
                    self.play(triangle.animate.align_on_border(UP))
                    self.play(square.animate.align_on_border(RIGHT))
                    self.play(star.animate.align_on_border(DOWN))
        """
        frame = (config.frame_x_radius, config.frame_y_radius, 0.0)
        source = self.get_anchor(direction)
        target: Point3D = np.sign(direction) * frame
        vector = target - source - buff * np.array(direction)
        vector = vector * abs(np.sign(direction))
        return self.translate(
            vector,
            **kwargs,
        )

    def align_to(
        self,
        point_or_mobject: Point3DLike | Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Aligns the object to a point or object in a certain direction.

        For example, ``mob.align_to((0.5, 1, 5), RIGHT)`` moves ``mob``
        horizontally so that its right edge is at x = 0.5, while
        ``mob.align_to(other_obj, UP)`` moves ``mob`` vertically so that its top
        edge lines up with ``other_obj``'s top edge.

        Parameters
        ----------
        point_or_mobject
            The point or mobject.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: AlignToExample

            class AlignToExample(Scene):
                def construct(self) -> None:
                    square = Square(side_length=1, color=RED).to_edge(DOWN)
                    triangle = Triangle(radius=2).move_to((2, 1, 0))

                    self.add(triangle)
                    self.play(square.animate.align_to(triangle, UP))
                    self.play(square.animate.align_to(triangle, DR))
        """
        if isinstance(point_or_mobject, Positionable):
            point_or_mobject = point_or_mobject.get_anchor(direction)

        all_points = self.get_all_points()
        vector = np.zeros(3)
        for dim in range(3):
            if direction[dim] != 0:
                source = self._get_extremum(all_points[:, dim], direction[dim])
                vector[dim] = point_or_mobject[dim] - source
        return self.translate(vector, **kwargs)

    def next_to(
        self,
        point_or_mobject: Point3DLike | Positionable,
        direction: Vector3DLike = RIGHT,
        *,
        aligned_edge: Vector3DLike = ORIGIN,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Moves the object next to a point or object.

        Parameters
        ----------
        point_or_mobject
            The point or mobject.
        direction
            The direction.
            Defaults to ``RIGHT``
        aligned_edge
            The edge to align.
            Defaults to ``ORIGIN``, meaning the object's center
        buff
            The distance to the point or mobject.
            Defaults to ``DEFAULT_MOBJECT_TO_MOBJECT_BUFFER``

        Returns
        -------
        Self
            The object itself.

        Example
        -------

        .. manim:: NextToExample
            :save_last_frame:

            class NextToExample(Scene):
                def construct(self):
                    dot = Dot()
                    circle = Circle()
                    square = Square()
                    triangle = Triangle()
                    dot.next_to(circle, RIGHT)
                    square.next_to(circle, LEFT)
                    triangle.next_to(circle, DOWN)
                    self.add(dot, circle, square, triangle)
        """
        direction = np.asarray(direction)
        aligned_edge = np.asarray(aligned_edge)
        if isinstance(point_or_mobject, Positionable):
            target_direction = aligned_edge + direction
            point_or_mobject = point_or_mobject.get_anchor(target_direction)
        source_direction = aligned_edge - direction
        source_point = self.get_anchor(source_direction)
        vector = point_or_mobject - source_point + buff * direction
        return self.translate(vector, **kwargs)

    def shift_onto_screen(
        self,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Moves the object onto the screen.

        Parameters
        ----------
        buff
            The distance to the border.
            Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

        Returns
        -------
        Self
            The object itself.
        """
        # TODO: Simplify implementation
        frame = (config.frame_x_radius, config.frame_y_radius)
        for dim, edge in (1, UP), (1, DOWN), (0, LEFT), (0, RIGHT):
            max_value = frame[dim] - buff
            edge_center = self.get_anchor(edge)
            if np.dot(edge_center, edge) > max_value:
                self.align_on_border(edge, buff=buff, **kwargs)
        return self

    def apply_function_to_position(
        self,
        function: Callable[[Point3D], Point3D],
        **kwargs: Any,
    ) -> Self:
        """Applies a function to the position of the object.

        Parameters
        ----------
        function
            The function to be applied.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_center(
            function(self.get_center()),
            **kwargs,
        )

    def is_off_screen(self) -> bool:
        """Whether the object is off screen.

        Returns
        -------
        bool
            Whether it's off screen.
        """
        all_points = self.get_all_points()
        return (
            self._get_extremum(all_points[:, 0], -1) > config.frame_x_radius
            or self._get_extremum(all_points[:, 0], 1) < -config.frame_x_radius
            or self._get_extremum(all_points[:, 1], -1) > config.frame_y_radius
            or self._get_extremum(all_points[:, 1], 1) < -config.frame_y_radius
        )

    def get_center_of_mass(self) -> Point3D:
        """Returns the mean of all points of the object.

        Returns
        -------
        Point3D
            The center of mass.
        """
        all_points = self.get_all_points()
        if len(all_points) == 0:
            return ORIGIN.copy()
        return np.array(
            [
                all_points[:, 0].mean(),
                all_points[:, 1].mean(),
                all_points[:, 2].mean(),
            ]
        )

    def get_boundary_point(self, direction: Vector3DLike) -> Point3D:
        """Returns the furthest point of the object in the given direction.

        .. note::
            Unlike anchor points, boundary points are guaranteed to lie on the convex hull of the object's points.

        Parameters
        ----------
        direction
            The direction of the anchor point.

        Returns
        -------
        Point3D
            The boundary point.

        Example
        -------
        .. manim:: GetBoundaryPointExample

            class GetBoundaryPointExample(Scene):
                def construct(self) -> None:
                    tracker = ValueTracker()
                    star = Star(n=7, inner_radius=1.5, outer_radius=3)

                    def get_vector() -> Vector3D:
                        value = tracker.get_value()
                        return np.array([np.cos(value), np.sin(value), 0.0])

                    dot = always_redraw(lambda: Dot(star.get_boundary_point(get_vector())))
                    arrow = always_redraw(lambda: Arrow().put_start_and_end_on(ORIGIN, get_vector()))

                    self.add(star, dot, arrow)
                    self.play(tracker.animate.set_value(2 * PI), run_time=3, rate_func=linear)

        """
        all_points = self.get_all_points()
        if len(all_points) == 0:
            return ORIGIN.copy()
        index = np.argmax(all_points.dot(direction))
        return all_points[index]

    def apply_function_to_submobject_positions(
        self,
        function: Callable[[Point3D], Point3D],
        **kwargs: Any,
    ) -> Self:
        """Applies a function to the submobject positions.

        Parameters
        ----------
        function
            The function to be applied.

        Returns
        -------
        Self
            The object itself.
        """
        raise NotImplementedError

    def space_out_submobjects(self, factor: float = 1.5) -> Self:
        """Scales the distance between the submobjects of the object.

        Parameters
        ----------
        factor
            The scaling factor.
            Defaults to ``1.5``
        about_point
            About which point to scale.
            Defaults to ``None``
        about_edge
            About which edge to scale.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: SpaceOutSubmobjectsExample

            class SpaceOutSubmobjectsExample(Scene):
                def construct(self) -> None:
                    circles = VGroup(
                        Circle(),
                        Circle(),
                        Circle(),
                        Circle(),
                    ).arrange_in_grid()
                    self.play(circles.animate.space_out_submobjects())
        """
        raise NotImplementedError

    def arrange(
        self,
        direction: Vector3DLike = RIGHT,
        *,
        aligned_edge: Vector3DLike = ORIGIN,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        center: bool = True,
        **kwargs: Any,
    ) -> Self:
        """Arranges the submobjects along a direction.

        Parameters
        ----------
        direction
            The direction of the anchor point.
            Defaults to ``RIGHT``
        aligned_edge
            The aligned edge.
            Defaults to ``ORIGIN``, meaning the object's center
        buff
            The distance between the objects.
            Defaults to ``DEFAULT_MOBJECT_TO_MOBJECT_BUFFER``
        center
            Whether to center the object after arranging.
            Defaults to ``True``

        Returns
        -------
        Self
            The object itself.

        Examples
        --------
        .. manim:: ArrangeExampleDots
            :save_last_frame:

            class ArrangeExampleDots(Scene):
                def construct(self):
                    dots = VGroup(
                        Dot().shift(
                            i * 0.1 * RIGHT * np.random.uniform(-1, 1)
                            + UP * np.random.uniform(-1, 1)
                        )
                        for i in range(0, 16)
                    ).set_color(BLUE)
                    self.play(dots.animate.arrange())


        .. manim:: ArrangeExampleBoxes
            :save_last_frame:

            class ArrangeExampleBoxes(Scene):
                def construct(self):
                    boxes = VGroup(
                        Square(color=RED),
                        Square(color=GREEN),
                        Square(color=BLUE),
                        Square(color=YELLOW),
                    )
                    self.play(boxes.animate.arrange(buff=1.0))

        .. manim:: ArrangeExampleShapes

            class ArrangeExampleShapes(Scene):
                def construct(self) -> None:
                    group = VGroup(
                        Rectangle(color=YELLOW),
                        Circle(),
                        Star(),
                    ).set_fill(opacity=1)

                    self.play(group.animate.arrange(RIGHT))
                    self.wait(0.1)
                    self.play(group.animate.arrange(DOWN, aligned_edge=RIGHT))

        """
        raise NotImplementedError

    def arrange_in_grid(
        self,
        rows: int | None = None,
        cols: int | None = None,
        *,
        buff: float | tuple[float, float] = MED_SMALL_BUFF,
        cell_alignment: Vector3DLike = ORIGIN,
        # TODO: replace with Vector3DLike
        row_alignments: str | None = None,
        # TODO: replace with Vector3DLike
        col_alignments: str | None = None,
        row_heights: Iterable[float | None] | None = None,
        col_widths: Iterable[float | None] | None = None,
        # TODO: replace with Vector3DLike
        flow_order: Literal["dr", "dl", "ur", "ul", "rd", "ld", "ru", "lu"] = "rd",
        **kwargs: Any,
    ) -> Self:
        """Arranges the submobjects in a grid.

        Parameters
        ----------
        rows
            The number of rows.
            Defaults to ``None``
        cols
            The number of columns.
            Defaults to ``None``
        buff
            The gap between grid cells.
            Defaults to ``MED_SMALL_BUFF``
        cell_alignment
            The way each submobject is aligned in its grid cell.
            Defaults to ``ORIGIN``, meaning the cell's center
        row_alignments
            The vertical alignment for each row.
            Defaults to ``None``
        col_alignments
            The horizontal alignment for each column.
            Defaults to ``None``
        row_heights
            Defines the heights for certain rows. For ``None``, the height is based on the highest element in that row.
            Defaults to ``None``
        col_widths
            Defines the widths for certain columns. For ``None``, the width is based on the widest element in that column.
            Defaults to ``None``
        flow_order
            The order in which submobjects fill the grid.
            Defaults to ``rd``, meaning first right and then down

        Returns
        -------
        Self
            The object itself.

        Raises
        ------
        ValueError
            If ``rows`` and ``cols`` are too small to fit all submobjects.
        ValueError
            If :code:`cols`, :code:`col_alignments` and :code:`col_widths` or :code:`rows`,
            :code:`row_alignments` and :code:`row_heights` have mismatching sizes.

        Notes
        -----
        If only one of ``cols`` and ``rows`` is set implicitly, the other one will be chosen big
        enough to fit all submobjects. If neither is set, they will be chosen to be about the same,
        tending towards ``cols`` > ``rows`` (simply because videos are wider than they are high).

        If both ``cell_alignment`` and ``row_alignments`` / ``col_alignments`` are defined, the latter has higher priority.


        Examples
        --------

        .. manim:: ArrangeInGridExampleBoxes
            :save_last_frame:

            class ArrangeInGridExampleBoxes(Scene):
                def construct(self):
                    boxes = VGroup(
                        Square(color=RED),
                        Square(color=GREEN),
                        Square(color=BLUE),
                        Square(color=YELLOW),
                        Square(color=PURPLE),
                        Square(color=ORANGE),
                    )
                    boxes.arrange_in_grid(rows=2, buff=0.1)
                    self.add(boxes)


        .. manim:: ArrangeInGridExampleNumbers
            :save_last_frame:

            class ArrangeInGridExampleNumbers(Scene):
                def construct(self):
                    boxes = VGroup(
                        Rectangle(WHITE, 0.5, 0.5).add(Text(str(i + 1)).scale(0.5))
                        for i in range(24)
                    )
                    boxes.arrange_in_grid(
                        buff=(0.25, 0.5),
                        col_alignments="lccccr",
                        row_alignments="uccd",
                        col_widths=[1, *[None] * 4, 1],
                        row_heights=[1, None, None, 1],
                        flow_order="dr",
                    )
                    self.add(boxes)

        """
        raise NotImplementedError

    # =========
    # endregion
    # =========

    # ===========
    # region SIZE
    # ===========
    def get_dim_size(self, dim: int) -> float:
        """Returns the size of the object along a certain dimension.

        Parameters
        ----------
        dim
            The dimension.

        Returns
        -------
        float
            The size along the dimension.
        """
        all_points = self.get_all_points()
        if len(all_points) == 0:
            return 0.0
        return np.ptp(all_points[:, dim])  # type: ignore[no-any-return]

    def scale_to_fit_dim(
        self,
        size: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that its size along a dimension is ``size``.

        Parameters
        ----------
        size
            The size.
        dim
            The dimension.
        about_point
            About which point to scale.
            Defaults to ``None``
        about_edge
            About which edge to scale.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_dim_size(dim)
        if source == 0:
            return self

        factor = size / source
        return self.scale(
            factor,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_width(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that its width is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to scale.
            Defaults to ``None``
        about_edge
            About which edge to scale.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.height
            np.float64(2.0)
            >>> sq.scale_to_fit_width(5)
            Square
            >>> sq.width
            np.float64(5.0)
            >>> sq.height
            np.float64(5.0)
        """
        return self.scale_to_fit_dim(
            size,
            0,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_height(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that its height is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to scale.
            Defaults to ``None``
        about_edge
            About which edge to scale.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        return self.scale_to_fit_dim(
            size,
            1,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_depth(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that its depth is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to scale.
            Defaults to ``None``
        about_edge
            About which edge to scale.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        return self.scale_to_fit_dim(
            size,
            2,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_size(
        self,
        size: Vector3DLike,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that its size is at most ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to scale.
            Defaults to ``None``
        about_edge
            About which edge to scale.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ScaleToFitSizeExample

            class ScaleToFitSizeExample(Scene):
                def construct(self) -> None:
                    square = Square(4)
                    circle = Ellipse().scale(6)
                    self.add_foreground_mobject(square)
                    self.play(circle.animate.scale_to_fit_size(square.size))
                    self.wait()
        """
        source = self.size
        if (source == 0).all():
            return self
        factor = min(size[dim] / source[dim] for dim in range(3) if source[dim] != 0)
        return self.scale(
            factor,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_dim(
        self,
        size: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that its size along a dimension is ``size``
        while leaving the other dimensions untouched.

        Parameters
        ----------
        size
            The size.
        dim
            The dimension.
        about_point
            About which point to stretch.
            Defaults to ``None``
        about_edge
            About which edge to stretch.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_dim_size(dim)
        if source == 0:
            return self
        factor = size / source
        return self.stretch(
            factor,
            dim,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_width(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that its width is ``size``
        while leaving the height and depth untouched.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to stretch.
            Defaults to ``None``
        about_edge
            About which edge to stretch.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.height
            np.float64(2.0)
            >>> sq.stretch_to_fit_width(5)
            Square
            >>> sq.width
            np.float64(5.0)
            >>> sq.height
            np.float64(2.0)
        """
        return self.stretch_to_fit_dim(
            size,
            0,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_height(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that its height is ``size``
        while leaving the width and depth untouched.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to stretch.
            Defaults to ``None``
        about_edge
            About which edge to stretch.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        ::

            >>> from manim import *
            >>> sq = Square()
            >>> sq.width
            np.float64(2.0)
            >>> sq.stretch_to_fit_height(5)
            Square
            >>> sq.height
            np.float64(5.0)
            >>> sq.width
            np.float64(2.0)
        """
        return self.stretch_to_fit_dim(
            size,
            1,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_depth(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that its depth is ``size``
        while leaving the width and height.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to stretch.
            Defaults to ``None``
        about_edge
            About which edge to stretch.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        return self.stretch_to_fit_dim(
            size,
            2,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_size(
        self,
        size: Vector3DLike,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that its size is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to stretch.
            Defaults to ``None``
        about_edge
            About which edge to stretch.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: StretchToFitSizeExample

            class StretchToFitSizeExample(Scene):
                def construct(self) -> None:
                    square = Square(4)
                    circle = Ellipse().scale(6)
                    self.add_foreground_mobject(square)
                    self.play(circle.animate.stretch_to_fit_size(square.size))
                    self.wait()
        """
        source = self.size
        if (source == 0).all():
            return self
        factor = (
            size[0] / source[0] if source[0] != 0 else 1,
            size[1] / source[1] if source[1] != 0 else 1,
            size[2] / source[2] if source[2] != 0 else 1,
        )
        return self.scale(
            factor,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # =========
    # endregion
    # =========

    # ===========
    # region MISC
    # ===========

    def flip(
        self,
        axis: Vector3DLike = UP,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Flips the object along an axis.

        Parameters
        ----------
        axis
            The axis to flip around.
            Defaults to ``UP``
        about_point
            About which point to flip.
            Defaults to ``None``
        about_edge
            About which edge to flip.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: FlipExample
            :save_last_frame:

            class FlipExample(Scene):
                def construct(self):
                    line = Line(LEFT, UR, color=RED)
                    flipped = line.copy().flip(axis=RIGHT).set_color(GREEN)
                    self.add(line, flipped)
        """
        return self.rotate(
            TAU / 2,
            axis,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def replace(
        self,
        other: Positionable,
        dim: int = 0,
        *,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        """TODO

        Parameters
        ----------
        other
            The other object.
        dim
            The dimension.
            Defaults to ``0``
        stretch
            Whether to stretch.
            Defaults to ``False``

        Returns
        -------
        Self
            The object itself.
        """
        # if not self.has_points() and not mobject.submobjects:
        #    raise Warning("Attempting to replace mobject with no points")
        if stretch:
            self.stretch_to_fit_width(other.width, **kwargs)
            self.stretch_to_fit_height(other.height, **kwargs)
            # TODO: add self.stretch_to_fit_depth(depth=mobject.get_depth(), **kwargs)
        else:
            self.scale_to_fit_dim(
                other.get_dim_size(dim),
                dim,
                **kwargs,
            )
        return self.set_center(other.get_center(), **kwargs)

    def surround(
        self,
        other: Positionable,
        dim: int = 0,
        *,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF,
        **kwargs: Any,
    ) -> Self:
        """Translates and resizes the object so that it surrounds the other object.

        Parameters
        ----------
        other
            The other object.
        dim
            The dimension.
            Defaults to ``0``
        stretch
            Whether to stretch (``True``) or scale (``False``).
            Defaults to ``False``
        buff
            The distance to the other object.
            Defaults to ``MED_SMALL_BUFF``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: SurroundExample

            class SurroundExample(Scene):
                def construct(self) -> None:
                    square = Square(side_length=1).to_corner(UL)
                    banner = ManimBanner()
                    self.add(banner)
                    self.play(square.animate.surround(banner, stretch=True))
        """
        # TODO: Avoid scaling/stretching twice
        self.replace(
            other,
            dim,
            stretch=stretch,
            **kwargs,
        )
        size = other.get_dim_size(dim)
        if size == 0:
            return self
        factor = (size + buff) / size
        return self.scale(factor, **kwargs)

    # endregion

    # ==============
    # region ALIASES
    # ==============

    # TODO: Only allow passing a single vector
    def shift(self, *vectors: Vector3DLike, **kwargs: Any) -> Self:
        """Translates the object by the given vectors.

        Note
        ----
        An alias for the :meth:`translate` method.

        Parameters
        ----------
        vectors
            The translation vectors.

        Returns
        -------
        Self
            The object itself.
        """
        vector: Vector3D
        if len(vectors) == 0:
            vector = ORIGIN
        elif len(vectors) == 1:
            vector = vectors[0]
        else:
            vector = reduce(operator.add, vectors)
        return self.translate(vector, **kwargs)

    def length_over_dim(self, dim: int) -> float:
        """Returns the size along a dimension of the object.

        Note
        ----
        An alias for the :meth:`get_dim_size` method.

        Parameters
        ----------
        dim
            The dimension.

        Returns
        -------
        float
            The size of the dimension.
        """
        return self.get_dim_size(dim)

    def get_critical_point(self, direction: Vector3DLike = ORIGIN) -> Point3D:
        """Returns the position of an anchor of the object.

        Note
        ----
        An alias for the :meth:`get_anchor` method.

        Parameters
        ----------
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Point3D
            The anchor position.

        Example
        -------
        .. code-block:: python

            sample = Arc(start_angle=PI / 7, angle=PI / 5)

            # These are all equivalent
            max_y_1 = sample.get_top()[1]
            max_y_2 = sample.get_critical_point(UP)[1]
            max_y_3 = sample.get_extremum_along_dim(dim=1, key=1)
        """
        return self.get_anchor(direction)

    def get_edge_center(self, direction: Vector3DLike) -> Point3D:
        """Returns an edge position of the object.

        Note
        ----
        An alias for the :meth:`get_anchor` method.

        Parameters
        ----------
        direction
            The direction of the anchor point.

        Returns
        -------
        Point3D
            The edge position.
        """
        return self.get_anchor(direction)

    def get_corner(self, direction: Vector3DLike) -> Point3D:
        """Returns the position of an anchor of the object.

        Note
        ----
        An alias for the :meth:`get_anchor` method.

        Parameters
        ----------
        direction
            The direction of the anchor point.

        Returns
        -------
        Point3D
            The anchor position.
        """
        return self.get_anchor(direction)

    def move_to(
        self,
        point_or_mobject: Point3DLike | Positionable,
        aligned_edge: Vector3DLike = ORIGIN,
        # coor_mask: Vector3DLike = np.array([1, 1, 1]),
        **kwargs: Any,
    ) -> Self:
        """Moves the object to the ``point`` or ``mobject``.

        Note
        ----
        An alias for the :meth:`set_anchor` and :meth:`match_anchor` methods.

        Parameters
        ----------
        point_or_mobject
            The point or mobject to move to.
        aligned_edge
            The aligned edge.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        if isinstance(point_or_mobject, Positionable):
            point_or_mobject = point_or_mobject.get_anchor(aligned_edge)
        return self.set_anchor(point_or_mobject, aligned_edge, **kwargs)

    def get_coord(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the ``dim``th coordinate at one of the object's anchor points.

        Note
        ----
        An alias for the :meth:`get_coordinate` method.

        Parameters
        ----------
        dim
            The dimension of the coordinate.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        float
            The coordinate value.
        """
        return self.get_coordinate(dim, direction)

    def set_coord(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the ``dim``th coordinate at the specified
        anchor point is at ``value``.

        Note
        ----
        An alias for the :meth:`set_coordinate` method.

        Parameters
        ----------
        value
            The coordinate value.
        dim
            The dimension of the coordinate.
        direction
            The direction of the anchor point.
            Defaults to ``ORIGIN``, meaning the object's center

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, dim, direction, **kwargs)

    def to_corner(
        self,
        corner: Vector3DLike = DL,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Moves the object to a corner.

        Note
        ----
        An alias for the :meth:`align_on_border` method.

        Parameters
        ----------
        corner
            The corner.
            Defaults to ``DL``, meaning the down left corner
        buff
            The distance to the corner.
            Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ToCornerExample
            :save_last_frame:

            class ToCornerExample(Scene):
                def construct(self):
                    circle = Circle()
                    circle.to_corner(UR)
                    tex = Tex("To the corner!")
                    mathtex = MathTex("x^3").shift(DOWN)
                    self.add(circle,tex,mathtex)
                    tex.to_corner(DL, buff=0)
                    mathtex.to_corner(UL, buff=1.5)
        """
        return self.align_on_border(corner, buff=buff, **kwargs)

    def to_edge(
        self,
        edge: Vector3DLike = LEFT,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Moves the object to a edge.

        Note
        ----
        An alias for the :meth:`align_on_border` method.

        Parameters
        ----------
        edge
            The edge.
            Defaults to ``LEFT``
        buff
            The distance to the edge.
            Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ToEdgeExample
            :save_last_frame:

            class ToEdgeExample(Scene):
                def construct(self):
                    tex_top = Tex("I am at the top!")
                    tex_top.to_edge(UP)
                    tex_side = Tex("I am moving to the side!")
                    circle = Circle().shift(2*DOWN)
                    self.add(tex_top, tex_side, circle)
                    tex_side.to_edge(LEFT)
                    circle.to_edge(RIGHT, buff=0)
        """
        return self.align_on_border(edge, buff=buff, **kwargs)

    @property
    def width(self) -> float:
        """The width of the object.

        Example
        -------
        .. manim:: WidthExample

            class WidthExample(Scene):
                def construct(self):
                    decimal = DecimalNumber().to_edge(UP)
                    rect = Rectangle(color=BLUE)
                    rect_copy = rect.copy().set_stroke(GRAY, opacity=0.5)

                    decimal.add_updater(lambda d: d.set_value(rect.width))

                    self.add(rect_copy, rect, decimal)
                    self.play(rect.animate.set(width=7))
                    self.wait()
        """
        return self.get_dim_size(0)

    @width.setter
    def width(self, value: float) -> None:
        self.scale_to_fit_width(value)

    @property
    def height(self) -> float:
        """The height of the object.

        Example
        -------
        .. manim:: HeightExample

            class HeightExample(Scene):
                def construct(self):
                    decimal = DecimalNumber().to_edge(UP)
                    rect = Rectangle(color=BLUE)
                    rect_copy = rect.copy().set_stroke(GRAY, opacity=0.5)

                    decimal.add_updater(lambda d: d.set_value(rect.height))

                    self.add(rect_copy, rect, decimal)
                    self.play(rect.animate.set(height=5))
                    self.wait()
        """
        return self.get_dim_size(1)

    @height.setter
    def height(self, value: float) -> None:
        self.scale_to_fit_height(value)

    @property
    def depth(self) -> float:
        """The depth of the object."""
        return self.get_dim_size(2)

    @depth.setter
    def depth(self, value: float) -> None:
        self.scale_to_fit_depth(value)

    @property
    def size(self) -> Vector3D:
        """The size of the object.

        Example
        -------
        .. manim:: SizeExample

            class SizeExample(Scene):
                def construct(self) -> None:
                    square = Square()
                    self.play(square.animate.set(size=(10, 0.25, 0)), run_time=3)
        """
        return np.array([self.width, self.height, self.depth])

    @size.setter
    def size(self, value: Vector3DLike) -> None:
        self.stretch_to_fit_size(value)

    # endregion

    # =================
    # region DEPRECATED
    # =================
    dim: int = 3

    # @deprecated(replacement="apply_points_function")
    def apply_points_function_about_point(
        self,
        func: Callable[[Point3D_Array], Point3D_Array],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        return self.apply_points_function(
            func,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # @deprecated(replacement="set_dim_size")
    def rescale_to_fit(
        self,
        size: float,
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        if stretch:
            return self.stretch_to_fit_dim(
                size,
                dim,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )
        else:
            return self.scale_to_fit_dim(
                size,
                dim,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )

    # @deprecated(replacement="stretch")
    def stretch_about_point(
        self,
        factor: float,
        dim: int,
        point: Point3DLike,
        **kwargs: Any,
    ) -> Self:
        return self.stretch(factor, dim, about_point=point, **kwargs)

    # @deprecated(replacement="get_coordinate")
    def get_extremum_along_dim(
        self,
        points: Point3DLike_Array | None = None,
        dim: int = 0,
        key: float = 0,
    ) -> float:
        if points is not None:
            points = np.asarray(points)
            return self._get_extremum(points[:, dim], key)  # type: ignore[call-overload]
        direction = np.zeros(3)
        direction[dim] = key
        return self.get_coordinate(dim, direction)

    # @deprecated(replacement="rotate")
    def rotate_about_origin(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        **kwargs: Any,
    ) -> Self:
        return self.rotate(
            angle,
            axis,
            about_point=ORIGIN,
            **kwargs,
        )

    # @deprecated()
    def reduce_across_dimension(
        self,
        reduce_func: Callable[[Iterable[float]], float],
        dim: int,
    ) -> float | None:
        all_points = self.get_all_points()
        if len(all_points) == 0:
            return None
        return reduce_func(all_points[:, dim])

    # @deprecated()
    def get_points_defining_boundary(self) -> Point3D_Array:
        return self.get_all_points()

    # @deprecated(replacement="arrange")
    def arrange_submobjects(
        self,
        direction: Vector3DLike = RIGHT,
        *,
        aligned_edge: Vector3DLike = ORIGIN,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        center: bool = True,
        **kwargs: Any,
    ) -> Self:
        return self.arrange(
            direction,
            aligned_edge=aligned_edge,
            buff=buff,
            center=center,
            **kwargs,
        )

    # @deprecated(replacement="repeat_points")
    def repeat(self, count: int, **kwargs: Any) -> Self:
        return self.repeat_points(count, **kwargs)

    # @deprecated(replacement="has_points")
    def has_no_points(self) -> bool:
        return not self.has_points()

    # @deprecated(replacement="rotate")
    def pose_at_angle(
        self,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.rotate(
            TAU / 14,
            UR,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # @deprecated(replacement="(scale|stretch)_to_fit_dim(other.get_dim_size())")
    def match_dim_size(
        self,
        other: Positionable,
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        if stretch:
            return self.stretch_to_fit_dim(
                other.get_dim_size(dim),
                dim,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )
        else:
            return self.scale_to_fit_dim(
                other.get_dim_size(dim),
                dim,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )

    # @deprecated(replacement="(scale|stretch)_to_fit_width(other.width)")
    def match_width(
        self,
        other: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.match_dim_size(
            other,
            0,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # @deprecated(replacement="(scale|stretch)_to_fit_height(other.height)")
    def match_height(
        self,
        other: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.match_dim_size(
            other,
            1,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # @deprecated(replacement="(scale|stretch)_to_fit_depth(other.depth)")
    def match_depth(
        self,
        other: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.match_dim_size(
            other,
            2,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # @deprecated(replacement="set_coordinate(other.get_coordinate())")
    def match_coord(
        self,
        other: Positionable,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.set_coordinate(
            other.get_coordinate(dim, direction), dim, direction, **kwargs
        )

    # @deprecated(replacement="set_x(other.get_x())")
    def match_x(
        self,
        other: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.set_x(other.get_x(direction), direction, **kwargs)

    # @deprecated(replacement="set_y(other.get_y())")
    def match_y(
        self,
        other: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.set_y(other.get_y(direction), direction, **kwargs)

    # @deprecated(replacement="set_z(other.get_z())")
    def match_z(
        self,
        other: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        return self.set_z(other.get_z(direction), direction, **kwargs)

    # =========
    # endregion
    # =========

    # ================
    # region UTILITIES
    # ================

    def _get_extremum(self, values: np.ndarray, key: float) -> float:
        if len(values) == 0:
            return 0.0
        return (  # type: ignore[no-any-return]
            values.min()
            if key < 0
            else (values.min() + values.max()) / 2
            if key == 0
            else values.max()
        )

    def throw_error_if_no_points(self) -> None:
        if not self.has_points():
            caller_name = sys._getframe(1).f_code.co_name
            cls = type(self).__name__
            message = f"Cannot call {cls}.{caller_name} because {self!r} has no points."
            pointful_family_members = [
                mob for mob in self.get_family() if mob.has_points()
            ]
            if pointful_family_members:
                count = len(pointful_family_members)
                message += (
                    f" Its family contains {count} "
                    f"mobject{'' if count == 1 else 's'} with points."
                )
            raise ValueError(message)

    # endregion
