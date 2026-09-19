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

    def set_points(self, points: Point3DLike_Array | Positionable) -> Self:
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

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: MatchPointsScene

            class MatchPointsScene(Scene):
                def construct(self):
                    circ = Circle(fill_color=RED, fill_opacity=0.8)
                    square = Square(fill_color=BLUE, fill_opacity=0.2)
                    self.add(circ)
                    self.wait(0.5)
                    self.play(circ.animate.match_points(square))
                    self.wait(0.5)
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
            The family.

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
        """Applies a function to the object.

        Parameters
        ----------
        function
            The function to be applied.
        should_skip
            A predicate function which returns ``True`` if a family member should be skipped. By default, family members with no points are skipped.

        Returns
        -------
        Self
            The object itself.
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
        default_point: Literal["ORIGIN", "OBJECT_CENTER"] = "OBJECT_CENTER",
        **kwargs: Any,
    ) -> Self:
        """Applies a function to the points of the object.

        Parameters
        ----------
        function
            The function to be applied.
        about_point
            About which point to apply the function., Defaults to ``None``
        about_edge
            About which edge to apply the function., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        about_point = self._get_about_point(about_point, about_edge, default_point)
        # TODO: Do we really need this?
        # Make a copy to prevent mutation of the original array if about_point is a view
        about_point = np.array(about_point, copy=True)

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
        """Applies a function to every point to the object.

        Parameters
        ----------
        function
            The function to be applied.
        about_point
            About which point to apply the function., Defaults to ``None``
        about_edge
            About which edge to apply the function., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.apply_points_function(
            lambda points: np.apply_along_axis(function, 1, points),
            about_point=about_point,
            about_edge=about_edge,
            default_point="ORIGIN",
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
        """Applies a complex function to every point to the object.

        Parameters
        ----------
        function
            The function to be applied.
        about_point
            About which point to apply the function., Defaults to ``None``
        about_edge
            About which edge to apply the function., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: ApplyFuncExample

            class ApplyFuncExample(Scene):
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
            The vector.

        Returns
        -------
        Self
            The object itself.


        .. note::
           Derived classes should override the :meth:`_translate` method for custom logic.
        """

        def apply(mob: Positionable) -> None:
            mob.points += vector

        return self.apply_to_family(apply, **kwargs)

    def scale(
        self,
        factor: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales the object uniformly by a factor along all dimensions.

        Parameters
        ----------
        factor
            The factor.
        about_point
            About which point to scale., Defaults to ``None``
        about_edge
            About which edge to scale., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. manim:: MobjectScaleExample
            :save_last_frame:

            class MobjectScaleExample(Scene):
                def construct(self):
                    f1 = Text("F")
                    f2 = Text("F").scale(2)
                    f3 = Text("F").scale(0.5)
                    f4 = Text("F").scale(-1)

                    vgroup = VGroup(f1, f2, f3, f4).arrange(6 * RIGHT)
                    self.add(vgroup)
        """

        def apply(points: Point3D_Array) -> Point3D_Array:
            return factor * points

        return self.apply_points_function(
            apply,
            about_point=about_point,
            about_edge=about_edge,
            default_point="OBJECT_CENTER",
            **kwargs,
        )

    def stretch(
        self,
        factor: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object by a factor along one dimension.

        Parameters
        ----------
        factor
            The factor.
        dim
            The dimension.
        about_point
            About which point to stretch., Defaults to ``None``
        about_edge
            About which edge to stretch., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """

        def apply(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        return self.apply_points_function(
            apply,
            about_point=about_point,
            about_edge=about_edge,
            default_point="OBJECT_CENTER",
            **kwargs,
        )

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Rotates the object by an angle along an axis.

        Parameters
        ----------
        angle
            The angle.
        axis
            The axis., Defaults to ``OUT``
        about_point
            About which point to rotate., Defaults to ``None``
        about_edge
            About which edge to rotate., Defaults to ``None``

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
        .. manim:: RotateMethodExample
            :save_last_frame:

            class RotateMethodExample(Scene):
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
        matrix = rotation_matrix(angle, axis)

        def apply(points: Point3D_Array) -> Point3D_Array:
            return points.dot(matrix.T)

        return self.apply_points_function(
            apply,
            about_point=about_point,
            about_edge=about_edge,
            default_point="OBJECT_CENTER",
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
        """Applies a transformation matrix to the points to the object.

        Parameters
        ----------
        matrix
            The matrix.
        about_point
            About which point to apply the matrix., Defaults to ``None``
        about_edge
            About which edge to apply the matrix., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        about_point = self._get_about_point(about_point, about_edge, "ORIGIN")
        matrix = np.asarray(matrix)
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
            default_point="ORIGIN",
            **kwargs,
        )

    def _get_about_point(
        self,
        about_point: Point3DLike | None,
        about_edge: Vector3DLike | None,
        default: Literal["ORIGIN", "OBJECT_CENTER"],
    ) -> Point3DLike:
        if about_point is None:
            if about_edge is None:
                if default == "ORIGIN":
                    return ORIGIN.copy()
                elif default == "OBJECT_CENTER":
                    return self.get_anchor(ORIGIN)
                else:
                    raise ValueError(default)
            else:
                return self.get_anchor(about_edge)
        else:
            return about_point

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
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point., i.e.

        Returns
        -------
        Point3D
            The position.
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
            The position.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_anchor(direction)
        vector = position - source
        return self.translate(vector, **kwargs)

    def match_anchor(
        self,
        other: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the anchor is identical to the corresponding anchor of the other object.

        Parameters
        ----------
        other
            The other object.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(other.get_anchor(direction), direction, **kwargs)

    def get_center(self) -> Point3D:
        """Returns the center position of the object.

        Returns
        -------
        Point3D
            The center position.
        """
        return self.get_anchor(ORIGIN)

    def set_center(
        self,
        position: Point3DLike,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the center position is at ``position``.

        Parameters
        ----------
        center
            The position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, ORIGIN, **kwargs)

    def match_center(self, other: Positionable, **kwargs: Any) -> Self:
        """Translates the object so that the center position is identical to the center of the other object.

        Parameters
        ----------
        other
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_center(other.get_center(), **kwargs)

    def center(self, **kwargs: Any) -> Self:
        """Translates the object so that the center position is at ``ORIGIN``.

        Returns
        -------
        Self
            The object itself.
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
        """Translates the object so that the top position is at ``position``.

        Parameters
        ----------
        position
            The position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, UP, **kwargs)

    def match_top(self, other: Positionable, **kwargs: Any) -> Self:
        """Translates the object so that the top position is identical to the top position of the other object.

        Parameters
        ----------
        other
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_top(other.get_top(), **kwargs)

    def get_bottom(self) -> Point3D:
        """Returns the bottom position of the object.

        Returns
        -------
        Point3D
            The bottom position.
        """
        return self.get_anchor(DOWN)

    def set_bottom(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that the bottom position is at ``position``.

        Parameters
        ----------
        position
            The position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, DOWN, **kwargs)

    def match_bottom(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Translates the object so that the bottom position is identical to the bottom position of the other object.

        Parameters
        ----------
        other
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_bottom(mobject.get_bottom(), **kwargs)

    def get_right(self) -> Point3D:
        """Returns the right position of the object.

        Returns
        -------
        Point3D
            The right position.
        """
        return self.get_anchor(RIGHT)

    def set_right(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that the right position is at ``position``.

        Parameters
        ----------
        position
            The position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, RIGHT, **kwargs)

    def match_right(self, other: Positionable, **kwargs: Any) -> Self:
        """Translates the object so that the right position is identical to the right position of the other object.

        Parameters
        ----------
        other
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_right(other.get_right(), **kwargs)

    def get_left(self) -> Point3D:
        """Returns the left position of the object.

        Returns
        -------
        Point3D
            The left position.
        """
        return self.get_anchor(LEFT)

    def set_left(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that the left position is at ``position``.

        Parameters
        ----------
        position
            The position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, LEFT, **kwargs)

    def match_left(self, other: Positionable, **kwargs: Any) -> Self:
        """Translates the object so that the left position is identical to the left position of the other object.

        Parameters
        ----------
        other
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_left(other.get_left(), **kwargs)

    def get_zenith(self) -> Point3D:
        """Returns the zenith position of the object.

        Returns
        -------
        Point3D
            The zenith position.
        """
        return self.get_anchor(OUT)

    def set_zenith(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that the zenith position is at ``position``.

        Parameters
        ----------
        position
            The position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, OUT, **kwargs)

    def match_zenith(self, other: Positionable, **kwargs: Any) -> Self:
        """Translates the object so that the zenith position is identical to the zenith position of the other object.

        Parameters
        ----------
        other
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_zenith(other.get_zenith(), **kwargs)

    def get_nadir(self) -> Point3D:
        """Returns the nadir position of the object.

        Returns
        -------
        Point3D
            The nadir position.
        """
        return self.get_anchor(IN)

    def set_nadir(self, position: Point3DLike, **kwargs: Any) -> Self:
        """Translates the object so that the nadir position is at ``position``.

        Parameters
        ----------
        position
            The position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position, IN, **kwargs)

    def match_nadir(self, other: Positionable, **kwargs: Any) -> Self:
        """Translates the object so that the nadir position is identical to the nadir position of the other object.

        Parameters
        ----------
        other
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_nadir(other.get_nadir(), **kwargs)

    def get_coordinate(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the coordinate of a dimension of the object.

        Parameters
        ----------
        dim
            The dimension.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        float
            The coordinate.
        """
        return self._get_extremum(self.get_all_points()[:, dim], direction[dim])

    def set_coordinate(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the coordinate of a dimension is at ``value``.

        Parameters
        ----------
        value
            The value.
        dim
            The dimension.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_coordinate(dim, direction)
        vector = np.zeros(3)
        vector[dim] = value - source
        return self.translate(vector, **kwargs)

    def match_coordinate(
        self,
        other: Positionable,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the coordinate of a dimension is identical to the corresponding coordinate of the other object.

        Parameters
        ----------
        other
            The other object.
        dim
            The dimension.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(
            other.get_coordinate(dim, direction),
            dim,
            direction,
            **kwargs,
        )

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the x coordinate of the object.

        Parameters
        ----------
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        float
            The x coordinate.
        """
        return self.get_coordinate(0, direction)

    def set_x(
        self, value: float, direction: Vector3DLike = ORIGIN, **kwargs: Any
    ) -> Self:
        """Translates the object so that the x coordinate is at ``value``.

        Parameters
        ----------
        value
            The value.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, 0, direction, **kwargs)

    def match_x(
        self,
        other: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the x coordinate is identical to the x coordinate of the other object.

        Parameters
        ----------
        other
            The other object.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_x(other.get_x(direction), direction, **kwargs)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the y coordinate of the object.

        Parameters
        ----------
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        float
            The y coordinate.
        """
        return self.get_coordinate(1, direction)

    def set_y(
        self, value: float, direction: Vector3DLike = ORIGIN, **kwargs: Any
    ) -> Self:
        """Translates the object so that the y coordinate is at ``value``.

        Parameters
        ----------
        value
            The value.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, 1, direction, **kwargs)

    def match_y(
        self,
        other: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the y coordinate is identical to the y coordinate of the other object.

        Parameters
        ----------
        other
            The other object.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_y(other.get_y(direction), direction, **kwargs)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the z coordinate of the object.

        Parameters
        ----------
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

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
        """Translates the object so that the z coordinate is at ``value``.

        Parameters
        ----------
        value
            The value.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, 2, direction, **kwargs)

    def match_z(
        self,
        other: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the z coordinate is identical to the z coordinate of the other object.

        Parameters
        ----------
        other
            The other object.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_z(other.get_z(direction), direction, **kwargs)

    def align_on_border(
        self,
        direction: Vector3DLike,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Aligns the object on a border.

        Parameters
        ----------
        direction
            The direction.
        buff
            The buff., Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

        Returns
        -------
        Self
            The object itself.
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
        """Aligns the object to a point.

        Parameters
        ----------
        point_or_mobject
            The point or mobject.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.

        Example
        -------
        .. code-block:: python

            # moves mob1 vertically so that its top edge lines ups with mob2's top edge
            mob1.align_to(mob2, UP)
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
        """Sets the position of the object next to a point.

        Parameters
        ----------
        point_or_mobject
            The point or mobject.
        direction
            The direction., Defaults to ``RIGHT``
        buff
            The buff., Defaults to ``DEFAULT_MOBJECT_TO_MOBJECT_BUFFER``
        aligned_edge
            The edge to align., Defaults to ``ORIGIN``

        Returns
        -------
        Self
            The object itself.

        Example
        -------

        .. manim:: GeometricShapes
            :save_last_frame:

            class GeometricShapes(Scene):
                def construct(self):
                    d = Dot()
                    c = Circle()
                    s = Square()
                    t = Triangle()
                    d.next_to(c, RIGHT)
                    s.next_to(c, LEFT)
                    t.next_to(c, DOWN)
                    self.add(d, c, s, t)
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
        """Shifts the object onto the screen.

        Parameters
        ----------
        buff
            The buff., Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

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
        return self.set_anchor(
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
        """Returns the center of mass of the object.

        Returns
        -------
        Point3D
            The center_of_mass.
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
        """Returns a boundary point of the object.

        Parameters
        ----------
        direction
            The direction.

        Returns
        -------
        Point3D
            The boundary point.
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

    def space_out_submobjects(
        self,
        factor: float = 1.5,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Spaces out the submobjects.

        Parameters
        ----------
        factor
            The factor., Defaults to ``1.5``
        about_point
            About which point to scale., Defaults to ``None``
        about_edge
            About which edge to scale., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
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
        """Arranges the submobject.

        Parameters
        ----------
        direction
            The direction., Defaults to ``RIGHT``
        aligned_edge
            The aligned edge., Defaults to ``ORIGIN``
        buff
            The buff., Defaults to ``DEFAULT_MOBJECT_TO_MOBJECT_BUFFER``
        center
            Whether to center the object after arranging., Defaults to ``True``

        Returns
        -------
        Self
            The object itself.

        Examples
        --------
        .. manim:: ArrangeExample
            :save_last_frame:

            class ArrangeExample(Scene):
                def construct(self):
                    s= VGroup(*[Dot().shift(i*0.1*RIGHT*np.random.uniform(-1,1)+UP*np.random.uniform(-1,1)) for i in range(0,15)])
                    s.shift(UP).set_color(BLUE)
                    s2= s.copy().set_color(RED)
                    s2.arrange()
                    s2.shift(DOWN)
                    self.add(s,s2)


        .. manim:: Example
            :save_last_frame:

            class Example(Scene):
                def construct(self):
                    s1 = Square()
                    s2 = Square()
                    s3 = Square()
                    s4 = Square()
                    x = VGroup(s1, s2, s3, s4).set_x(0).arrange(buff=1.0)
                    self.add(x)
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
            The number of rows., Defaults to ``None``
        cols
            The number of columns., Defaults to ``None``
        buff
            The gap between grid cells., Defaults to ``MED_SMALL_BUFF``
        cell_alignment
            The way each submobject is aligned in its grid cell., Defaults to ``ORIGIN``
        row_alignments
            The vertical alignment for each row., Defaults to ``None``
        col_alignments
            The horizontal alignment for each column., Defaults to ``None``
        row_heights
            Defines the heights for certain rows. For ``None``, the height is based on the highest element in that row., Defaults to ``None``
        col_widths
            Defines the widths for certain columns. For ``None``, the width is based on the widest element in that column., Defaults to ``None``
        flow_order
            The order in which submobjects fill the grid., Defaults to ``rd``, meaning first right and then down.

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

        .. manim:: ExampleBoxes
            :save_last_frame:

            class ExampleBoxes(Scene):
                def construct(self):
                    boxes=VGroup(*[Square() for s in range(0,6)])
                    boxes.arrange_in_grid(rows=2, buff=0.1)
                    self.add(boxes)


        .. manim:: ArrangeInGrid
            :save_last_frame:

            class ArrangeInGrid(Scene):
                def construct(self):
                    boxes = VGroup(*[
                        Rectangle(WHITE, 0.5, 0.5).add(Text(str(i+1)).scale(0.5))
                        for i in range(24)
                    ])
                    self.add(boxes)

                    boxes.arrange_in_grid(
                        buff=(0.25,0.5),
                        col_alignments="lccccr",
                        row_alignments="uccd",
                        col_widths=[1, *[None]*4, 1],
                        row_heights=[1, None, None, 1],
                        flow_order="dr"
                    )
        """
        raise NotImplementedError

    # =========
    # endregion
    # =========

    # ===========
    # region SIZE
    # ===========
    def get_dim_size(self, dim: int) -> float:
        """Returns the size of a dimension of the object.

        Parameters
        ----------
        dim
            The dimension.

        Returns
        -------
        float
            The dim size.
        """
        all_points = self.get_all_points()
        if len(all_points) == 0:
            return 0.0
        return np.ptp(all_points[:, dim])  # type: ignore[no-any-return]

    def set_dim_size(
        self,
        size: float,
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizes the object so that the size of a dimension is ``size``.

        Parameters
        ----------
        size
            The size.
        dim
            The dimension.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to resize., Defaults to ``None``
        about_edge
            About which edge to resize., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_dim_size(dim)
        if source == 0:
            return self
        factor = size / source
        if stretch:
            return self.stretch(
                factor,
                dim,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )
        else:
            return self.scale(
                factor,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )

    def match_dim_size(
        self,
        other: Positionable,
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizes the object so that the size of a dimension is identical to the corresponding size of the other object.

        Parameters
        ----------
        other
            The other object.
        dim
            The dimension.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to set the dim size., Defaults to ``None``
        about_edge
            About which edge to set the dim size., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            other.get_dim_size(dim),
            dim,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def get_width(self) -> float:
        """Returns the width of the object.

        Returns
        -------
        float
            The width.
        """
        return self.get_dim_size(0)

    def set_width(
        self,
        size: float,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizes the object so that the width is ``size``.

        Parameters
        ----------
        size
            The size.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to set the width., Defaults to ``None``
        about_edge
            About which edge to set the width., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            0,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def match_width(
        self,
        other: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizes the object so that the width is identical to the width of the other object.

        Parameters
        ----------
        other
            The other object.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to set the width., Defaults to ``None``
        about_edge
            About which edge to set the width., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_width(
            other.get_width(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def get_height(self) -> float:
        """Returns the height of the object.

        Returns
        -------
        float
            The height.
        """
        return self.get_dim_size(1)

    def set_height(
        self,
        size: float,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizes the object so that the height is ``size``.

        Parameters
        ----------
        size
            The size.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to set the height., Defaults to ``None``
        about_edge
            About which edge to set the height., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            1,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def match_height(
        self,
        other: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizse the object so that the height is identical to the height of the other object.

        Parameters
        ----------
        other
            The other object.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to set the width., Defaults to ``None``
        about_edge
            About which edge to set the width., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_height(
            other.get_height(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def get_depth(self) -> float:
        """Returns the depth of the object.

        Returns
        -------
        float
            The depth.
        """
        return self.get_dim_size(2)

    def set_depth(
        self,
        size: float,
        stretch: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizes the object so that the depth is ``size``.

        Parameters
        ----------
        size
            The size.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to set the depth., Defaults to ``None``
        about_edge
            About which edge to set the depth., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            2,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def match_depth(
        self,
        other: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Resizes the object so that the depth is identical to the depth of the other object.

        Parameters
        ----------
        other
            The other object.
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        about_point
            About which point to set the width., Defaults to ``None``
        about_edge
            About which edge to set the width., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_depth(
            other.get_depth(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_dim(
        self,
        size: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that the size of a dimension is ``size``.

        Parameters
        ----------
        size
            The size.
        dim
            The dimension.
        about_point
            About which point to scale., Defaults to ``None``
        about_edge
            About which edge to scale., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            dim,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_width(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that the width is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to scale., Defaults to ``None``
        about_edge
            About which edge to scale., Defaults to ``None``

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
        return self.set_dim_size(
            size,
            0,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_height(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that the height is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to scale., Defaults to ``None``
        about_edge
            About which edge to scale., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            1,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_depth(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales the object so that the depth is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to scale., Defaults to ``None``
        about_edge
            About which edge to scale., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            2,
            stretch=False,
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
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that the size of a dimension is ``size``.

        Parameters
        ----------
        size
            The size.
        dim
            The dimension.
        about_point
            About which point to stretch., Defaults to ``None``
        about_edge
            About which edge to stretch., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            dim,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_width(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that the width is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to stretch., Defaults to ``None``
        about_edge
            About which edge to stretch., Defaults to ``None``

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
        return self.set_dim_size(
            size,
            0,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_height(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that the height is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to stretch., Defaults to ``None``
        about_edge
            About which edge to stretch., Defaults to ``None``

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
            >>> sq.scale_to_fit_height(5)
            Square
            >>> sq.height
            np.float64(5.0)
            >>> sq.width
            np.float64(5.0)
        """
        return self.set_dim_size(
            size,
            1,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_depth(
        self,
        size: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches the object so that the depth is ``size``.

        Parameters
        ----------
        size
            The size.
        about_point
            About which point to stretch., Defaults to ``None``
        about_edge
            About which edge to stretch., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size,
            2,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # endregion

    ##########################
    ########## MISC ##########
    ##########################

    def flip(
        self,
        axis: Vector3DLike = UP,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Flips the object along an axis.

        Parameters
        ----------
        axis
            The axis to flip around., Defaults to ``UP``
        about_point
            About which point to flip., Defaults to ``None``
        about_edge
            About which edge to flip., Defaults to ``None``

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
                    s= Line(LEFT, RIGHT+UP).shift(4*LEFT)
                    self.add(s)
                    s2= s.copy().flip()
                    self.add(s2)
        """
        return self.rotate(
            TAU / 2,
            axis,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def pose_at_angle(
        self,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Poses the object at an angle.

        Parameters
        ----------
        about_point
            About which point to pose., Defaults to ``None``
        about_edge
            About which edge to pose., Defaults to ``None``

        Returns
        -------
        Self
            The object itself.
        """
        return self.rotate(
            TAU / 14,
            UR,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def replace(
        self,
        mobject: Positionable,
        dim: int = 0,
        *,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        """TODO

        Parameters
        ----------
        mobject
            The mobject.
        dim
            The dimension., Defaults to ``0``
        stretch
            Whether to stretch., Defaults to ``False``

        Returns
        -------
        Self
            The object itself.
        """
        # if not self.has_points() and not mobject.submobjects:
        #    raise Warning("Attempting to replace mobject with no points")
        if stretch:
            self.stretch_to_fit_width(mobject.get_width(), **kwargs)
            self.stretch_to_fit_height(mobject.get_height(), **kwargs)
            # TODO: add self.stretch_to_fit_depth(depth=mobject.get_depth(), **kwargs)
        else:
            self.scale_to_fit_dim(
                mobject.get_dim_size(dim),
                dim,
                **kwargs,
            )
        return self.set_center(mobject.get_center(), **kwargs)

    def surround(
        self,
        mobject: Positionable,
        dim: int = 0,
        *,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF,
        **kwargs: Any,
    ) -> Self:
        """Translates and resizes the object so that it surrounds the other object.

        Parameters
        ----------
        mobject
            The mobject.
        dim
            The dimension., Defaults to ``0``
        stretch
            Whether to stretch (``True``) or scale (``False``)., Defaults to ``False``
        buff
            The buff., Defaults to ``MED_SMALL_BUFF``

        Returns
        -------
        Self
            The object itself.
        """
        # TODO: Avoid scaling/stretching twice
        self.replace(
            mobject,
            dim,
            stretch=stretch,
            **kwargs,
        )
        size = mobject.get_dim_size(dim)
        if size == 0:
            return self
        factor = (size + buff) / size
        return self.scale(factor, **kwargs)

    #############################
    ########## ALIASES ##########
    #############################

    # TODO: Only allow passing a single vector
    def shift(self, *vectors: Vector3DLike, **kwargs: Any) -> Self:
        """Translates the object by the given vectors.

        Note
        ----
        An alias for the :meth:`translate` method.

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
        """Returns the size of a dimension of the object.

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
            The dim size.
        """
        return self.get_dim_size(dim)

    def get_critical_point(self, direction: Vector3DLike = ORIGIN) -> Point3D:
        """Returns a critical point of the object.

        Note
        ----
        An alias for the :meth:`get_anchor` method.

        Parameters
        ----------
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Point3D
            The critical point.

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
            The direction.

        Returns
        -------
        Point3D
            The edge position.
        """
        return self.get_anchor(direction)

    def get_corner(self, direction: Vector3DLike) -> Point3D:
        """Returns a corner position of the object.

        Note
        ----
        An alias for the :meth:`get_anchor` method.

        Parameters
        ----------
        direction
            The direction.

        Returns
        -------
        Point3D
            The corner position.
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
        An alias for the :meth:`set_anchor` method.

        Parameters
        ----------
        point_or_mobject
            The point or mobject.
        aligned_edge
            The aligned edge., Defaults to ``ORIGIN``

        Returns
        -------
        Self
            The object itself.
        """
        if not isinstance(point_or_mobject, Positionable):
            return self.set_anchor(point_or_mobject, aligned_edge, **kwargs)

        else:
            return self.match_anchor(point_or_mobject, aligned_edge, **kwargs)

    def get_coord(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the coordinate of a dimension of the object.

        Note
        ----
        An alias for the :meth:`get_coordinate` method.

        Parameters
        ----------
        dim
            The dimension.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        float
            The coordinate.
        """
        return self.get_coordinate(dim, direction)

    def set_coord(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the coordinate of a dimension is at coordinate.

        Note
        ----
        An alias for the :meth:`set_coordinate` method.

        Parameters
        ----------
        value
            The value.
        dim
            The dimension.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(value, dim, direction, **kwargs)

    def match_coord(
        self,
        other: Positionable,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Translates the object so that the coordinate of a dimension is identical to the corresponding coordinate of the other object.

        Note
        ----
        An alias for the :meth:`match_coordinate` method.

        Parameters
        ----------
        other
            The other object.
        dim
            The dimension.
        direction
            The direction., Defaults to ``ORIGIN``, i.e. the object's center point.

        Returns
        -------
        Self
            The object itself.
        """
        return self.match_coordinate(other, dim, direction, **kwargs)

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
            The corner., Defaults to ``DL``
        buff
            The buff., Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

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
                    c = Circle()
                    c.to_corner(UR)
                    t = Tex("To the corner!")
                    t2 = MathTex("x^3").shift(DOWN)
                    self.add(c,t,t2)
                    t.to_corner(DL, buff=0)
                    t2.to_corner(UL, buff=1.5)
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
            The edge., Defaults to ``LEFT``
        buff
            The buff., Defaults to ``DEFAULT_MOBJECT_TO_EDGE_BUFFER``

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
                    c = Circle().shift(2*DOWN)
                    self.add(tex_top, tex_side, c)
                    tex_side.to_edge(LEFT)
                    c.to_edge(RIGHT, buff=0)
        """
        return self.align_on_border(edge, buff=buff, **kwargs)

    @property
    def width(self) -> float:
        """The width of the object.

        A property for the :meth:`get_width` and :meth:`set_width` methods.

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
        return self.get_width()

    @width.setter
    def width(self, value: float) -> None:
        self.set_width(value)

    @property
    def height(self) -> float:
        """The height of the object.

        A property for the :meth:`get_height` and :meth:`set_height` methods.

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
        return self.get_height()

    @height.setter
    def height(self, value: float) -> None:
        self.set_height(value)

    @property
    def depth(self) -> float:
        """The depth of the object.

        A property for the :meth:`get_depth` and :meth:`set_depth` methods.
        """
        return self.get_depth()

    @depth.setter
    def depth(self, value: float) -> None:
        self.set_depth(value)

    ################################
    ########## DEPRECATED ##########
    ################################
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
        length: float,
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        return self.set_dim_size(
            length,
            dim,
            stretch=stretch,
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

    ###############################
    ########## UTILITIES ##########
    ###############################
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
