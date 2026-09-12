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
    """A positionable object."""

    # =============
    # region POINTS
    # =============

    points: Point3D_Array = np.zeros((0, 3))

    def get_all_points(self) -> Point3D_Array:
        """Returns all points.

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
        """Sets the points.

        Does not affect family members.

        Parameters
        ----------
        points : Point3DLike_Array
            The points.

        Returns
        -------
        Self
            The object itself.
        """
        self.points = np.asarray(points, dtype=float, copy=True)
        return self

    def match_points(self, mobject: Positionable, *, strict: bool = False) -> Self:
        """Matches the points.

        The points of the family members are matched in order.

        Parameters
        ----------
        other : Positionable
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
        for sm1, sm2 in zip(self.get_family(), mobject.get_family(), strict=strict):
            sm1.points = sm2.points.copy()
        return self

    def reset_points(self, **kwargs: Any) -> Self:
        """Resets the points.

        Does not affect family members.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_points(np.zeros((0, 3)), **kwargs)

    def reverse_points(self, **kwargs: Any) -> Self:
        """Reverses the points.

        Returns
        -------
        Self
            The object itself.
        """

        def apply(mob: Positionable) -> None:
            mob.points[:] = mob.points[::-1]

        return self.apply_to_family(func=apply, **kwargs)

    # TODO: Rename to `repeat_points`
    def repeat(self, count: int, **kwargs: Any) -> Self:
        """Repeats the points.

        Can make transition animations nicer.

        Parameters
        ----------
        count : int
            The repeat count.

        Returns
        -------
        Self
            The object itself.
        """

        def apply(mob: Positionable) -> None:
            mob.points = np.tile(mob.points, (count, 1))

        return self.apply_to_family(func=apply, **kwargs)

    def get_num_points(self) -> int:
        """Returns the number of points.

        Does not take family members into account.

        Returns
        -------
        int
            The number of points.
        """
        return len(self.points)

    def has_no_points(self) -> bool:
        """Whether this has no points.

        Does not take family members into account.

        Returns
        -------
        bool
            Has no points.
        """
        return len(self.points) == 0

    def has_points(self) -> bool:
        """Whether this has points.

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
        """Returns the family.

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
        # TODO: Rename to `function`
        func: Callable[[Positionable], Any],
        *,
        should_skip: Callable[[Positionable], bool] = lambda mob: mob.has_no_points(),
        **kwargs: Any,
    ) -> Self:
        """Applies a function.

        Parameters
        ----------
        func : Callable[[Positionable], Any]
            The function.
        should_skip : Callable -> bool, optional
            Whether a family member should be skipped., by default `has_no_points()`

        Returns
        -------
        Self
            The object itself.
        """
        for mob in self.get_family():
            if not should_skip(mob):
                func(mob)

        return self

    def apply_points_function(
        self,
        # TODO: Rename to `function`
        func: Callable[[Point3D_Array], Point3D_Array],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a points function.

        Parameters
        ----------
        func : Callable[[Point3D_Array], Point3D_Array]
            The function.
        about_point : Point3DLike | None, optional
            About which point to apply the function., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to apply the function., by default None

        Returns
        -------
        Self
            The object itself.
        """
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_anchor(about_edge)
        else:
            # TODO: Is this required?
            # Make a copy to prevent mutation of the original array if about_point is a view
            about_point = np.array(about_point, copy=True)

        def apply(mob: Positionable) -> None:
            mob.points -= about_point
            mob.points = func(mob.points)
            mob.points += about_point

        return self.apply_to_family(
            func=apply,
            **kwargs,
        )

    # TODO: Rename to `apply_point_function`?
    def apply_function(
        self,
        function: Callable[[Point3D], Point3D],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a point function.

        Parameters
        ----------
        function : Callable[[Point3D], Point3D]
            The function.
        about_point : Point3DLike | None, optional
            About which point to apply the function., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to apply the function., by default None

        Returns
        -------
        Self
            The object itself.
        """
        # Default to applying matrix about the origin, not mobjects center
        if about_point is None and about_edge is None:
            about_point = ORIGIN

        return self.apply_points_function(
            func=lambda points: np.apply_along_axis(function, 1, points),
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
        """Applies a complex function.

        Parameters
        ----------
        function : Callable[[complex], complex]
            The function.
        about_point : Point3DLike | None, optional
            About which point to apply the function., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to apply the function., by default None

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
            function=R3_func,
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
        """Applies a translation.

        Parameters
        ----------
        vector : Vector3DLike
            The vector.

        Returns
        -------
        Self
            The object itself.
        """
        return self.apply_to_family(
            func=lambda mob: mob.points.__iadd__(vector),
            **kwargs,
        )

    def scale(
        self,
        # TODO: Rename to `factor`
        scale_factor: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a uniform scaling.

        Parameters
        ----------
        scale_factor : float
            The scale_factor.
        about_point : Point3DLike | None, optional
            About which point to scale., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to scale., by default None

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
        return self.apply_points_function(
            lambda points: points.__imul__(scale_factor),
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
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a non-uniform scaling.

        Parameters
        ----------
        factor : float
            The factor.
        dim : int
            The dimension.
        about_point : Point3DLike | None, optional
            About which point to stretch., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to stretch., by default None

        Returns
        -------
        Self
            The object itself.
        """

        def func(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        return self.apply_points_function(
            func=func,
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
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Applies a rotation.

        Parameters
        ----------
        angle : float
            The angle.
        axis : Vector3DLike, optional
            The axis., by default OUT
        about_point : Point3DLike | None, optional
            About which point to rotate., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to rotate., by default None

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
        if about_point is None and about_edge is None:
            about_edge = ORIGIN
        matrix = rotation_matrix(angle, axis)
        return self.apply_matrix(
            matrix=matrix,
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
        """Applies a transformation matrix.

        Parameters
        ----------
        matrix : MatrixMN
            The matrix.
        about_point : Point3DLike | None, optional
            About which point to apply the matrix., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to apply the matrix., by default None

        Returns
        -------
        Self
            The object itself.
        """
        # Default to applying matrix about the origin, not mobjects center
        if about_point is None and about_edge is None:
            about_point = ORIGIN
        matrix = np.asarray(matrix)
        if matrix.shape == (3, 3):
            full_matrix = matrix
        else:
            full_matrix = np.identity(3)
            full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        return self.apply_points_function(
            func=lambda points: np.dot(points, full_matrix.T),
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    # =========
    # endregion
    # =========

    # ===============
    # region POSITION
    # ===============

    def get_anchor(self, direction: Vector3DLike = ORIGIN) -> Point3D:
        """Returns the position of an anchor.

        Parameters
        ----------
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Point3D
            The position.
        """
        all_points = self.get_all_points()
        return np.array(
            [
                self._get_extremum(all_points[:, dim], key=key)
                for dim, key in enumerate(direction)
            ]
        )

    def set_anchor(
        self,
        position: Point3DLike,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Sets the position of an anchor.

        Parameters
        ----------
        position : Point3DLike
            The position.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_anchor(direction=direction)
        vector = position - source
        return self.translate(vector=vector, **kwargs)

    def match_anchor(
        self,
        mobject: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Matches the anchor position.

        Parameters
        ----------
        other : Positionable
            The other object.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(
            position=mobject.get_anchor(direction=direction),
            direction=direction,
            **kwargs,
        )

    def get_center(self) -> Point3D:
        """Returns the center position.

        Returns
        -------
        Point3D
            The center position.
        """
        return self.get_anchor(direction=ORIGIN)

    def set_center(
        self,
        center: Point3DLike,
        **kwargs: Any,
    ) -> Self:
        """Sets the center position.

        Parameters
        ----------
        center : Point3DLike
            The center position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position=center, direction=ORIGIN, **kwargs)

    def match_center(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Matches the center position.

        Parameters
        ----------
        other : Positionable
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_center(center=mobject.get_center(), **kwargs)

    def center(self, **kwargs: Any) -> Self:
        """Sets the position to the origin.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_center(center=ORIGIN, **kwargs)

    def get_top(self) -> Point3D:
        """Returns the top position.

        Returns
        -------
        Point3D
            The top position.
        """
        return self.get_anchor(direction=UP)

    def set_top(self, top: Point3DLike, **kwargs: Any) -> Self:
        """Sets the top position.

        Parameters
        ----------
        top : Point3DLike
            The top position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position=top, direction=UP, **kwargs)

    def match_top(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Matches the top position.

        Parameters
        ----------
        other : Positionable
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_top(top=mobject.get_top(), **kwargs)

    def get_bottom(self) -> Point3D:
        """Returns the bottom position.

        Returns
        -------
        Point3D
            The bottom position.
        """
        return self.get_anchor(direction=DOWN)

    def set_bottom(self, bottom: Point3DLike, **kwargs: Any) -> Self:
        """Sets the bottom position.

        Parameters
        ----------
        bottom : Point3DLike
            The bottom position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position=bottom, direction=DOWN, **kwargs)

    def match_bottom(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Matches the bottom position.

        Parameters
        ----------
        other : Positionable
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_bottom(bottom=mobject.get_bottom(), **kwargs)

    def get_right(self) -> Point3D:
        """Returns the right position.

        Returns
        -------
        Point3D
            The right position.
        """
        return self.get_anchor(direction=RIGHT)

    def set_right(self, right: Point3DLike, **kwargs: Any) -> Self:
        """Sets the right position.

        Parameters
        ----------
        right : Point3DLike
            The right position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position=right, direction=RIGHT, **kwargs)

    def match_right(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Matches the right position.

        Parameters
        ----------
        other : Positionable
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_right(right=mobject.get_right(), **kwargs)

    def get_left(self) -> Point3D:
        """Returns the left position.

        Returns
        -------
        Point3D
            The left position.
        """
        return self.get_anchor(direction=LEFT)

    def set_left(self, left: Point3DLike, **kwargs: Any) -> Self:
        """Sets the left position.

        Parameters
        ----------
        left : Point3DLike
            The left position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position=left, direction=LEFT, **kwargs)

    def match_left(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Matches the left position.

        Parameters
        ----------
        other : Positionable
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_left(left=mobject.get_left(), **kwargs)

    def get_zenith(self) -> Point3D:
        """Returns the zenith position.

        Returns
        -------
        Point3D
            The zenith position.
        """
        return self.get_anchor(direction=OUT)

    def set_zenith(self, zenith: Point3DLike, **kwargs: Any) -> Self:
        """Sets the zenith position.

        Parameters
        ----------
        zenith : Point3DLike
            The zenith position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position=zenith, direction=OUT, **kwargs)

    def match_zenith(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Matches the zenith position.

        Parameters
        ----------
        other : Positionable
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_zenith(zenith=mobject.get_zenith(), **kwargs)

    def get_nadir(self) -> Point3D:
        """Returns the nadir position.

        Returns
        -------
        Point3D
            The nadir position.
        """
        return self.get_anchor(direction=IN)

    def set_nadir(self, nadir: Point3DLike, **kwargs: Any) -> Self:
        """Sets the nadir position.

        Parameters
        ----------
        nadir : Point3DLike
            The nadir position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_anchor(position=nadir, direction=IN, **kwargs)

    def match_nadir(self, mobject: Positionable, **kwargs: Any) -> Self:
        """Matches the nadir position.

        Parameters
        ----------
        other : Positionable
            The other object.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_nadir(nadir=mobject.get_nadir(), **kwargs)

    def get_coordinate(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the coordinate of a dimension.

        Parameters
        ----------
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        float
            The coordinate.
        """
        return self._get_extremum(
            values=self.get_all_points()[:, dim],
            key=direction[dim],
        )

    def set_coordinate(
        self,
        coordinate: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Sets the coordinate of a dimension.

        Parameters
        ----------
        coordinate : float
            The coordinate.
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_coordinate(dim=dim, direction=direction)
        vector = np.zeros(3)
        vector[dim] = coordinate - source
        return self.translate(vector=vector, **kwargs)

    def match_coordinate(
        self,
        mobject: Positionable,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Matches the coordinate of a dimension.

        Parameters
        ----------
        other : Positionable
            The other object.
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(
            coordinate=mobject.get_coordinate(dim=dim, direction=direction),
            dim=dim,
            direction=direction,
            **kwargs,
        )

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the x coordinate.

        Parameters
        ----------
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        float
            The x coordinate.
        """
        return self.get_coordinate(dim=0, direction=direction)

    def set_x(self, x: float, direction: Vector3DLike = ORIGIN, **kwargs: Any) -> Self:
        """Sets the x coordinate.

        Parameters
        ----------
        x : float
            The x coordinate.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(coordinate=x, dim=0, direction=direction, **kwargs)

    def match_x(
        self,
        mobject: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Matches the x coordinate.

        Parameters
        ----------
        other : Positionable
            The other object.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_x(
            x=mobject.get_x(direction=direction),
            direction=direction,
            **kwargs,
        )

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the x coordinate.

        Parameters
        ----------
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        float
            The y coordinate.
        """
        return self.get_coordinate(dim=1, direction=direction)

    def set_y(self, y: float, direction: Vector3DLike = ORIGIN, **kwargs: Any) -> Self:
        """Sets the y coordinate.

        Parameters
        ----------
        y : float
            The y coordinate.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(
            coordinate=y,
            dim=1,
            direction=direction,
            **kwargs,
        )

    def match_y(
        self,
        mobject: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Matches the y coordinate.

        Parameters
        ----------
        other : Positionable
            The other object.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_y(
            y=mobject.get_y(direction=direction),
            direction=direction,
            **kwargs,
        )

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the z coordinate.

        Parameters
        ----------
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        float
            The z coordinate.
        """
        return self.get_coordinate(dim=2, direction=direction)

    def set_z(
        self,
        z: float,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Sets the z coordinate.

        Parameters
        ----------
        z : float
            The z coordinate.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(
            coordinate=z,
            dim=2,
            direction=direction,
            **kwargs,
        )

    def match_z(
        self,
        mobject: Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Matches the z coordinate.

        Parameters
        ----------
        other : Positionable
            The other object.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_z(
            z=mobject.get_z(direction=direction),
            direction=direction,
            **kwargs,
        )

    def align_on_border(
        self,
        direction: Vector3DLike,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Aligns itself on the border.

        Parameters
        ----------
        direction : Vector3DLike
            The direction.
        buff : float, optional
            The buff., by default `DEFAULT_MOBJECT_TO_EDGE_BUFFER`

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
        # TODO: Rename to `point`
        mobject_or_point: Point3DLike | Positionable,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Aligns itself to a point.

        Parameters
        ----------
        mobject_or_point : Point3DLike | Positionable
            The point.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

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
        if isinstance(mobject_or_point, Positionable):
            mobject_or_point = mobject_or_point.get_anchor(direction=direction)

        all_points = self.get_all_points()
        vector = np.zeros(3)
        for dim in range(3):
            if direction[dim] != 0:
                source = self._get_extremum(all_points[:, dim], key=direction[dim])
                vector[dim] = mobject_or_point[dim] - source
        return self.translate(
            vector=vector,
            **kwargs,
        )

    def next_to(
        self,
        # TODO: Rename to `point`
        mobject_or_point: Point3DLike | Positionable,
        direction: Vector3DLike = RIGHT,
        *,
        aligned_edge: Vector3DLike = ORIGIN,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Sets the position next to a point.

        Parameters
        ----------
        mobject_or_point : Point3DLike | Positionable
            The point.
        direction : Vector3DLike, optional
            The direction., by default RIGHT
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
        aligned_edge : Vector3DLike, optional
            The edge to align., by default ORIGIN

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
        if isinstance(mobject_or_point, Positionable):
            target_direction = aligned_edge + direction
            mobject_or_point = mobject_or_point.get_anchor(direction=target_direction)
        source_direction = aligned_edge - direction
        source_point = self.get_anchor(direction=source_direction)
        vector = mobject_or_point - source_point + buff * direction
        return self.translate(
            vector=vector,
            **kwargs,
        )

    def shift_onto_screen(
        self,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Shifts onto the screen.

        Parameters
        ----------
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER

        Returns
        -------
        Self
            The object itself.
        """
        # TODO: Simplify implementation
        frame = (config.frame_x_radius, config.frame_y_radius)
        for dim, edge in (1, UP), (1, DOWN), (0, LEFT), (0, RIGHT):
            max_value = frame[dim] - buff
            edge_center = self.get_anchor(direction=edge)
            if np.dot(edge_center, edge) > max_value:
                self.align_on_border(
                    direction=edge,
                    buff=buff,
                    **kwargs,
                )
        return self

    def apply_function_to_position(
        self,
        function: Callable[[Point3D], Point3D],
        **kwargs: Any,
    ) -> Self:
        """Applies a function to the position.

        Parameters
        ----------
        function : Callable[[Point3D], Point3D]
            The function.

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
        """Whether this is off screen.

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
        """Returns the center of mass.

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
        """Returns a boundary point.

        Parameters
        ----------
        direction : Vector3DLike
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
        function : Callable[[Point3D], Point3D]
            The function.

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
        factor : float, optional
            The factor., by default 1.5
        about_point : Point3DLike | None, optional
            About which point to scale., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to scale., by default None

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
        direction : Vector3DLike, optional
            The direction., by default RIGHT
        aligned_edge : Vector3DLike, optional
            The aligned edge., by default ORIGIN
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_MOBJECT_BUFFER
        center : bool, optional
            Whether to center., by default True

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
        rows : int | None, optional
            The number of rows., by default None
        cols : int | None, optional
            The number of columns., by default None
        buff : float | tuple[float, float], optional
            The gap between grid cells., by default MED_SMALL_BUFF
        cell_alignment : Vector3DLike, optional
            The way each submobject is aligned in its grid cell., by default ORIGIN
        row_alignments : Literal['u', 'c', 'd'] | None, optional
            The vertical alignment for each row., by default None
        col_alignments : Literal['l', 'c', 'r'] | None, optional
            The horizontal alignment for each column., by default None
        row_heights : Iterable[float  |  None] | None, optional
            Defines the heights for certain rows. For ``None``, the height is based on the highest element in that row., by default None
        col_widths : Iterable[float  |  None] | None, optional
            Defines the widths for certain columns. For ``None``, the width is based on the widest element in that column., by default None
        flow_order : Literal['dr', 'dl', 'ur', 'ul', 'rd', 'ld', 'ru', 'lu'], optional
            The order in which submobjects fill the grid., by default "rd"

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
        """Returns the size of a dimension.

        Parameters
        ----------
        dim : int
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
        """Sets the size of a dimension.

        Parameters
        ----------
        size : float
            The size.
        dim : int
            The dimension.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the dim size., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the dim size., by default None

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_dim_size(dim=dim)
        if source == 0:
            return self
        factor = size / source
        if stretch:
            return self.stretch(
                factor=factor,
                dim=dim,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )
        else:
            return self.scale(
                scale_factor=factor,
                about_point=about_point,
                about_edge=about_edge,
                **kwargs,
            )

    def match_dim_size(
        self,
        mobject: Positionable,
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Matches the size of a dimension.

        Parameters
        ----------
        other : Positionable
            The other object.
        dim : int
            The dimension.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the dim size., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the dim size., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=mobject.get_dim_size(dim=dim),
            dim=dim,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def get_width(self) -> float:
        """Returns the width.

        Returns
        -------
        float
            The width.
        """
        return self.get_dim_size(dim=0)

    def set_width(
        self,
        width: float,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Sets the width.

        Parameters
        ----------
        width : float
            The width.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the width., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the width., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=width,
            dim=0,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def match_width(
        self,
        mobject: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Matches the width.

        Parameters
        ----------
        other : Positionable
            The other object.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the width., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the width., by default None

        Returns
        -------
        Self
            _description_
        """
        return self.set_width(
            width=mobject.get_width(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def get_height(self) -> float:
        """Returns the height.

        Returns
        -------
        float
            The height.
        """
        return self.get_dim_size(dim=1)

    def set_height(
        self,
        height: float,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Sets the height.

        Parameters
        ----------
        height : float
            The height.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the height., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the height., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=height,
            dim=1,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def match_height(
        self,
        mobject: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Matches the height.

        Parameters
        ----------
        other : Positionable
            The other object.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the width., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the width., by default None

        Returns
        -------
        Self
            _description_
        """
        return self.set_height(
            height=mobject.get_height(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def get_depth(self) -> float:
        """Returns the depth.

        Returns
        -------
        float
            The depth.
        """
        return self.get_dim_size(dim=2)

    def set_depth(
        self,
        depth: float,
        stretch: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Sets the depth.

        Parameters
        ----------
        depth : float
            The depth.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the depth., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the depth., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=depth,
            dim=2,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def match_depth(
        self,
        mobject: Positionable,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Matches the depth.

        Parameters
        ----------
        other : Positionable
            The other object.
        stretch : bool, optional
            Whether to stretch or scale., by default False
        about_point : Point3DLike | None, optional
            About which point to set the width., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to set the width., by default None

        Returns
        -------
        Self
            _description_
        """
        return self.set_depth(
            depth=mobject.get_depth(),
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
        """Scales to fit a size of a dimension.

        Parameters
        ----------
        size : float
            The size.
        dim : int
            The dimension.
        about_point : Point3DLike | None, optional
            About which point to scale., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to scale., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=size,
            dim=dim,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_width(
        self,
        width: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales to fit a width.

        Parameters
        ----------
        width : float
            The width.
        about_point : Point3DLike | None, optional
            About which point to scale., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to scale., by default None

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
            size=width,
            dim=0,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_height(
        self,
        height: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales to fit a height.

        Parameters
        ----------
        height : float
            The height.
        about_point : Point3DLike | None, optional
            About which point to scale., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to scale., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=height,
            dim=1,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def scale_to_fit_depth(
        self,
        depth: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Scales to fit a depth.

        Parameters
        ----------
        depth : float
            The depth.
        about_point : Point3DLike | None, optional
            About which point to scale., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to scale., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=depth,
            dim=2,
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
        """Stretches to fit a size of a dimension.

        Parameters
        ----------
        size : float
            The size.
        dim : int
            The dimension.
        about_point : Point3DLike | None, optional
            About which point to stretch., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to stretch., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=size,
            dim=dim,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_width(
        self,
        width: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches to fit a width.

        Parameters
        ----------
        width : float
            The width.
        about_point : Point3DLike | None, optional
            About which point to stretch., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to stretch., by default None

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
            size=width,
            dim=0,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_height(
        self,
        height: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches to fit a height.

        Parameters
        ----------
        height : float
            The height.
        about_point : Point3DLike | None, optional
            About which point to stretch., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to stretch., by default None

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
            size=height,
            dim=1,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def stretch_to_fit_depth(
        self,
        depth: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """Stretches to fit a depth.

        Parameters
        ----------
        depth : float
            The depth.
        about_point : Point3DLike | None, optional
            About which point to stretch., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to stretch., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_dim_size(
            size=depth,
            dim=2,
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
        """Flips.

        Parameters
        ----------
        axis : Vector3DLike, optional
            The axis., by default UP
        about_point : Point3DLike | None, optional
            About which point to flip., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to flip., by default None

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
            angle=TAU / 2,
            axis=axis,
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
        """Poses at an angle.

        Parameters
        ----------
        about_point : Point3DLike | None, optional
            About which point to pose., by default None
        about_edge : Vector3DLike | None, optional
            About which edge to pose., by default None

        Returns
        -------
        Self
            The object itself.
        """
        return self.rotate(
            angle=TAU / 14,
            axis=UR,
            about_point=about_point,
            about_edge=about_edge,
            **kwargs,
        )

    def replace(
        self,
        mobject: Positionable,
        # TODO: rename to `dim`
        dim_to_match: int = 0,
        *,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        """

        Parameters
        ----------
        mobject : Positionable
            The mobject.
        dim_to_match : int, optional
            The dimension., by default 0
        stretch : bool, optional
            Whether to stretch., by default False

        Returns
        -------
        Self
            The object itself.
        """
        # if self.has_no_points() and not mobject.submobjects:
        #    raise Warning("Attempting to replace mobject with no points")
        if stretch:
            self.stretch_to_fit_width(width=mobject.get_width(), **kwargs)
            self.stretch_to_fit_height(height=mobject.get_height(), **kwargs)
            # TODO: add self.stretch_to_fit_depth(depth=mobject.get_depth(), **kwargs)
        else:
            self.scale_to_fit_dim(
                size=mobject.get_dim_size(dim=dim_to_match),
                dim=dim_to_match,
                **kwargs,
            )
        return self.set_center(center=mobject.get_center(), **kwargs)

    def surround(
        self,
        mobject: Positionable,
        # TODO: Rename to `dim`
        dim_to_match: int = 0,
        *,
        stretch: bool = False,
        buff: float = MED_SMALL_BUFF,
        **kwargs: Any,
    ) -> Self:
        """Surrounds an object.

        Parameters
        ----------
        mobject : Positionable
            The mobject.
        dim_to_match : int, optional
            The dimension., by default 0
        stretch : bool, optional
            Whether to stretch or scale., by default False
        buff : float, optional
            The buff., by default MED_SMALL_BUFF

        Returns
        -------
        Self
            The object itself.
        """
        # TODO: Avoid scaling/stretching twice
        self.replace(
            mobject=mobject,
            dim_to_match=dim_to_match,
            stretch=stretch,
            **kwargs,
        )
        size = mobject.get_dim_size(dim=dim_to_match)
        if size == 0:
            return self
        factor = (size + buff) / size
        return self.scale(
            scale_factor=factor,
            **kwargs,
        )

    #############################
    ########## ALIASES ##########
    #############################

    # TODO: Only allow passing a single vector
    def shift(self, *vectors: Vector3DLike, **kwargs: Any) -> Self:
        """Applies a translation.

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
        return self.translate(vector=vector, **kwargs)

    def length_over_dim(self, dim: int) -> float:
        """Returns the size of a dimension.

        Note
        ----
        An alias for the :meth:`get_dim_size` method.

        Parameters
        ----------
        dim : int
            The dimension.

        Returns
        -------
        float
            The dim size.
        """
        return self.get_dim_size(dim=dim)

    def get_critical_point(self, direction: Vector3DLike = ORIGIN) -> Point3D:
        """Returns a critical point.

        Note
        ----
        An alias for the :meth:`get_position` method.

        Parameters
        ----------
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

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
        return self.get_anchor(direction=direction)

    def get_edge_center(self, direction: Vector3DLike) -> Point3D:
        """Returns an edge position.

        Note
        ----
        An alias for the :meth:`get_position` method.

        Parameters
        ----------
        direction : Vector3DLike
            The direction.

        Returns
        -------
        Point3D
            The edge position.
        """
        return self.get_anchor(direction=direction)

    def get_corner(self, direction: Vector3DLike) -> Point3D:
        """Returns a corner position.

        Note
        ----
        An alias for the :meth:`get_position` method.

        Parameters
        ----------
        direction : Vector3DLike
            The direction.

        Returns
        -------
        Point3D
            The corner position.
        """
        return self.get_anchor(direction=direction)

    def move_to(
        self,
        point_or_mobject: Point3DLike | Positionable,
        aligned_edge: Vector3DLike = ORIGIN,
        # coor_mask: Vector3DLike = np.array([1, 1, 1]),
        **kwargs: Any,
    ) -> Self:
        """Sets the position.

        Note
        ----
        An alias for the :meth:`set_position` method.

        Parameters
        ----------
        point_or_mobject : Point3DLike | Positionable
            The point_or_mobject.
        aligned_edge : Vector3DLike, optional
            The aligned edge., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        if not isinstance(point_or_mobject, Positionable):
            return self.set_anchor(
                position=point_or_mobject,
                direction=aligned_edge,
                **kwargs,
            )

        else:
            return self.match_anchor(
                mobject=point_or_mobject,
                direction=aligned_edge,
                **kwargs,
            )

    def get_coord(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        """Returns the coordinate of a dimension.

        Note
        ----
        An alias for the :meth:`get_coordinate` method.

        Parameters
        ----------
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        float
            The coordinate.
        """
        return self.get_coordinate(dim=dim, direction=direction)

    def set_coord(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Sets the coordinate of a dimension.

        Note
        ----
        An alias for the :meth:`set_coordinate` method.

        Parameters
        ----------
        value : float
            The value.
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(
            coordinate=value,
            dim=dim,
            direction=direction,
            **kwargs,
        )

    def match_coord(
        self,
        mobject: Positionable,
        dim: int,
        direction: Vector3DLike = ORIGIN,
        **kwargs: Any,
    ) -> Self:
        """Matches the coordinate of a dimension.

        Note
        ----
        An alias for the :meth:`match_coordinate` method.

        Parameters
        ----------
        mobject : Positionable
            The other object.
        dim : int
            The dimension.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.match_coordinate(
            mobject=mobject,
            dim=dim,
            direction=direction,
            **kwargs,
        )

    def to_corner(
        self,
        corner: Vector3DLike = DL,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Sets the position to a corner.

        Note
        ----
        An alias for the :meth:`align_on_border` method.

        Parameters
        ----------
        corner : Vector3DLike, optional
            The corner., by default DL
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER

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
        return self.align_on_border(direction=corner, buff=buff, **kwargs)

    def to_edge(
        self,
        edge: Vector3DLike = LEFT,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        **kwargs: Any,
    ) -> Self:
        """Sets the position to an edge.

        Note
        ----
        An alias for the :meth:`align_on_border` method.

        Parameters
        ----------
        edge : Vector3DLike, optional
            The edge., by default LEFT
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER

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
        return self.align_on_border(direction=edge, buff=buff, **kwargs)

    @property
    def width(self) -> float:
        """The width.

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
        self.set_width(width=value)

    @property
    def height(self) -> float:
        """The height.

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
        self.set_height(height=value)

    @property
    def depth(self) -> float:
        """The depth.

        A property for the :meth:`get_depth` and :meth:`set_depth` methods.
        """
        return self.get_depth()

    @depth.setter
    def depth(self, value: float) -> None:
        self.set_depth(depth=value)

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
            func=func,
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
            size=length,
            dim=dim,
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
        return self.stretch(factor=factor, dim=dim, about_point=point, **kwargs)

    # @deprecated(replacement="get_coordinate")
    def get_extremum_along_dim(
        self,
        points: Point3DLike_Array | None = None,
        dim: int = 0,
        key: float = 0,
    ) -> float:
        if points is not None:
            points = np.asarray(points)
            return self._get_extremum(values=points[:, dim], key=key)  # type: ignore[call-overload]
        direction = np.zeros(3)
        direction[dim] = key
        return self.get_coordinate(dim=dim, direction=direction)

    # @deprecated(replacement="rotate")
    def rotate_about_origin(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        **kwargs: Any,
    ) -> Self:
        return self.rotate(
            angle=angle,
            axis=axis,
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
            direction=direction,
            aligned_edge=aligned_edge,
            buff=buff,
            center=center,
            **kwargs,
        )

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
        if self.has_no_points():
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
