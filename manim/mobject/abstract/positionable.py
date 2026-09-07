from __future__ import annotations

import operator
import sys
from collections.abc import Callable, Iterable
from functools import reduce
from typing import Any, Self

import numpy as np

from manim._config import config
from manim.constants import (
    DEFAULT_MOBJECT_TO_EDGE_BUFFER,
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
    """

    An abstract positionable object.

    Attributes
    ----------
    points: :class:`Point3D_Array`


    See Also
    --------
    :class:`~manim.Mobject`
    """

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

        If an array is passed, set's the points of this object.
        If another object is passed, matches the points of corresponding family members in order.

        Parameters
        ----------
        points : Point3DLike_Array | Positionable
            The points.

        Returns
        -------
        Self
            The object itself.
        """
        if isinstance(points, Positionable):
            for sm1, sm2 in zip(self.get_family(), points.get_family(), strict=False):
                sm1.points = np.array(sm2.points)
        else:
            self.points = np.array(points, dtype=float)
        return self

    def reset_points(self) -> Self:
        """Resets the points.

        Does not affect family members.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_points(np.zeros((0, 3)))

    def reverse_points(self) -> Self:
        """Reverses the points.

        Returns
        -------
        Self
            The object itself.
        """

        def apply(mob: Positionable) -> None:
            mob.points[:] = mob.points[::-1]

        return self.apply_to_family(func=apply)

    def repeat(self, count: int) -> Self:
        """Repeats the points.

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

        return self.apply_to_family(apply)

    def get_num_points(self) -> int:
        """The number of points.

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
        """
        return [self]

    def apply_to_family(
        self,
        # TODO: Rename to `function`
        func: Callable[[Positionable], Any],
        *,
        should_skip: Callable[[Positionable], bool] = lambda mob: mob.has_no_points(),
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
            about_point = self.get_position(about_edge)
        else:
            # TODO: Is this required?
            # Make a copy to prevent mutation of the original array if about_point is a view
            about_point = np.array(about_point, copy=True)

        def apply(mob: Positionable) -> None:
            mob.points -= about_point
            mob.points = func(mob.points)
            mob.points += about_point

        return self.apply_to_family(func=apply)

    # TODO: Rename to `apply_point_function`?
    def apply_function(
        self,
        function: Callable[[Point3D], Point3D],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
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
        )

    def apply_complex_function(
        self,
        function: Callable[[complex], complex],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
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
        """

        def R3_func(point: Point3D) -> Point3D:
            x, y, z = point
            xy_complex = function(complex(x, y))
            return np.array([xy_complex.real, xy_complex.imag, z])

        return self.apply_function(
            function=R3_func,
            about_point=about_point,
            about_edge=about_edge,
        )

    # =========
    # endregion
    # =========

    # ======================
    # region TRANSFORMATIONS
    # ======================

    def translate(self, vector: Vector3DLike) -> Self:
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
        return self.apply_to_family(func=lambda mob: mob.points.__iadd__(vector))

    def scale(
        self,
        # TODO: Rename to `factor`
        scale_factor: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
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
        """
        return self.apply_points_function(
            lambda points: points.__imul__(scale_factor),
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch(
        self,
        factor: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
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
        )

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
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
        """
        matrix = rotation_matrix(angle, axis).T
        return self.apply_points_function(
            func=lambda points: np.dot(points, matrix),
            about_point=about_point,
            about_edge=about_edge,
        )

    def apply_matrix(
        self,
        matrix: MatrixMN,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
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
        full_matrix = np.identity(3)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        return self.apply_points_function(
            func=lambda points: np.dot(points, full_matrix.T),
            about_point=about_point,
            about_edge=about_edge,
        )

    # =========
    # endregion
    # =========

    # ===============
    # region POSITION
    # ===============

    def get_position(self, direction: Vector3DLike = ORIGIN) -> Point3D:
        """Returns the position.

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

    def set_position(
        self,
        position: Point3DLike | Positionable,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        """Sets the position.

        Parameters
        ----------
        position : Point3DLike | Positionable
            The position.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        source = self.get_position(direction=direction)
        if isinstance(position, Positionable):
            position = position.get_position(direction=direction)
        vector = position - source
        return self.translate(vector=vector)

    def get_center(self) -> Point3D:
        """Returns the center position.

        Returns
        -------
        Point3D
            The center position.
        """
        return self.get_position(direction=ORIGIN)

    def set_center(self, center: Point3DLike | Positionable) -> Self:
        """Sets the center position.

        Parameters
        ----------
        center : Point3DLike | Positionable
            The center position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_position(position=center, direction=ORIGIN)

    def center(self) -> Self:
        """Sets the position to the origin.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_center(center=ORIGIN)

    def get_top(self) -> Point3D:
        """Returns the top position.

        Returns
        -------
        Point3D
            The top position.
        """
        return self.get_position(direction=UP)

    def set_top(self, top: Point3DLike | Positionable) -> Self:
        """Sets the top position.

        Parameters
        ----------
        top : Point3DLike | Positionable
            The top position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_position(position=top, direction=UP)

    def get_bottom(self) -> Point3D:
        """Returns the bottom position.

        Returns
        -------
        Point3D
            The bottom position.
        """
        return self.get_position(direction=DOWN)

    def set_bottom(self, bottom: Point3DLike | Positionable) -> Self:
        """Sets the bottom.

        Parameters
        ----------
        bottom : Point3DLike | Positionable
            The bottom position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_position(position=bottom, direction=DOWN)

    def get_right(self) -> Point3D:
        """Returns the right.

        Returns
        -------
        Point3D
            The right position.
        """
        return self.get_position(direction=RIGHT)

    def set_right(self, right: Point3DLike | Positionable) -> Self:
        """Sets the right position.

        Parameters
        ----------
        right : Point3DLike | Positionable
            The right position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_position(position=right, direction=RIGHT)

    def get_left(self) -> Point3D:
        """Returns the left position.

        Returns
        -------
        Point3D
            The left position.
        """
        return self.get_position(direction=LEFT)

    def set_left(self, left: Point3DLike | Positionable) -> Self:
        """Sets the left position.

        Parameters
        ----------
        left : Point3DLike | Positionable
            The left position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_position(position=left, direction=LEFT)

    def get_zenith(self) -> Point3D:
        """Returns the zenith position.

        Returns
        -------
        Point3D
            The zenith position.
        """
        return self.get_position(direction=OUT)

    def set_zenith(self, zenith: Point3DLike | Positionable) -> Self:
        """Sets the zenith position.

        Parameters
        ----------
        zenith : Point3DLike | Positionable
            The zenith position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_position(position=zenith, direction=OUT)

    def get_nadir(self) -> Point3D:
        """Returns the nadir position.

        Returns
        -------
        Point3D
            The nadir position.
        """
        return self.get_position(direction=IN)

    def set_nadir(self, nadir: Point3DLike | Positionable) -> Self:
        """Sets the nadir position.

        Parameters
        ----------
        nadir : Point3DLike | Positionable
            The nadir position.

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_position(position=nadir, direction=IN)

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
        coordinate: float | Positionable,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        """Sets the coordinate of a dimension.

        Parameters
        ----------
        coordinate : float | Positionable
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
        if isinstance(coordinate, Positionable):
            coordinate = coordinate.get_coordinate(dim=dim, direction=direction)
        vector = np.zeros(3)
        vector[dim] = coordinate - source
        return self.translate(vector=vector)

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

    def set_x(self, x: float | Positionable, direction: Vector3DLike = ORIGIN) -> Self:
        """Sets the x coordinate.

        Parameters
        ----------
        x : float | Positionable
            The x coordinate.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(coordinate=x, dim=0, direction=direction)

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

    def set_y(self, y: float | Positionable, direction: Vector3DLike = ORIGIN) -> Self:
        """Sets the y coordinate.

        Parameters
        ----------
        y : float | Positionable
            The y coordinate.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(coordinate=y, dim=1, direction=direction)

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

    def set_z(self, z: float | Positionable, direction: Vector3DLike = ORIGIN) -> Self:
        """Sets the z coordinate.

        Parameters
        ----------
        z : float | Positionable
            The z coordinate.
        direction : Vector3DLike, optional
            The direction., by default ORIGIN

        Returns
        -------
        Self
            The object itself.
        """
        return self.set_coordinate(coordinate=z, dim=2, direction=direction)

    def align_on_border(
        self,
        direction: Vector3DLike,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Aligns itself on the border.

        Parameters
        ----------
        direction : Vector3DLike
            The direction.
        buff : float, optional
            The buff., by default DEFAULT_MOBJECT_TO_EDGE_BUFFER

        Returns
        -------
        Self
            The object itself.
        """
        frame = (config.frame_x_radius, config.frame_y_radius, 0.0)
        source = self.get_position(direction)
        target: Point3D = np.sign(direction) * frame
        vector = target - source - buff * np.array(direction)
        vector = vector * abs(np.sign(direction))
        return self.translate(vector)

    def align_to(
        self,
        # TODO: Rename to `point`
        mobject_or_point: Point3DLike | Positionable,
        direction: Vector3DLike = ORIGIN,
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
        """
        if isinstance(mobject_or_point, Positionable):
            mobject_or_point = mobject_or_point.get_position(direction=direction)

        all_points = self.get_all_points()
        vector = np.zeros(3)
        for dim in range(3):
            if direction[dim] != 0:
                source = self._get_extremum(all_points[:, dim], key=direction[dim])
                vector[dim] = mobject_or_point[dim] - source
        return self.translate(vector=vector)

    def shift_onto_screen(
        self,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Shifts itself onto the screen.

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
            edge_center = self.get_position(direction=edge)
            if np.dot(edge_center, edge) > max_value:
                self.align_on_border(direction=edge, buff=buff)
        return self

    def apply_function_to_position(
        self,
        function: Callable[[Point3D], Point3D],
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
        return self.set_position(function(self.get_center()))

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
            return ORIGIN
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
            return ORIGIN
        index = np.argmax(all_points.dot(direction))
        return all_points[index]

    # TODO: next_to

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
        size: float | Positionable,
        dim: int,
        stretch: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the size of a dimension.

        Parameters
        ----------
        size : float | Positionable
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
        if isinstance(size, Positionable):
            size = size.get_dim_size(dim=dim)
        factor = size / source
        if stretch:
            return self.stretch(
                factor=factor,
                dim=dim,
                about_point=about_point,
                about_edge=about_edge,
            )
        else:
            return self.scale(
                scale_factor=factor,
                about_point=about_point,
                about_edge=about_edge,
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
        width: float | Positionable,
        stretch: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the width.

        Parameters
        ----------
        width : float | Positionable
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
        height: float | Positionable,
        stretch: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the height.

        Parameters
        ----------
        height : float | Positionable
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
        depth: float | Positionable,
        stretch: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Sets the depth.

        Parameters
        ----------
        depth : float | Positionable
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
        )

    def scale_to_fit_dim(
        self,
        size: float | Positionable,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a size of a dimension.

        Parameters
        ----------
        size : float | Positionable
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
        )

    def scale_to_fit_width(
        self,
        width: float | Positionable,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a width.

        Parameters
        ----------
        width : float | Positionable
            The width.
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
            size=width,
            dim=0,
            stretch=False,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit_height(
        self,
        height: float | Positionable,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a height.

        Parameters
        ----------
        height : float | Positionable
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
        )

    def scale_to_fit_depth(
        self,
        depth: float | Positionable,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Scales to fit a depth.

        Parameters
        ----------
        depth : float | Positionable
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
        )

    def stretch_to_fit_dim(
        self,
        size: float | Positionable,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit a size of a dimension.

        Parameters
        ----------
        size : float | Positionable
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
        )

    def stretch_to_fit_width(
        self,
        width: float | Positionable,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit a width.

        Parameters
        ----------
        width : float | Positionable
            The width.
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
            size=width,
            dim=0,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_height(
        self,
        height: float | Positionable,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit a height.

        Parameters
        ----------
        height : float | Positionable
            The height.
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
            size=height,
            dim=1,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_depth(
        self,
        depth: float | Positionable,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        """Stretches to fit a depth.

        Parameters
        ----------
        depth : float | Positionable
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
        """
        return self.rotate(
            angle=TAU / 2,
            axis=axis,
            about_point=about_point,
            about_edge=about_edge,
        )

    def pose_at_angle(
        self,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
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
        )

    def replace(
        self,
        mobject: Positionable,
        # TODO: rename to `dim`
        dim_to_match: int = 0,
        stretch: bool = False,
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
            self.stretch_to_fit_width(width=mobject)
            self.stretch_to_fit_height(height=mobject)
            # TODO: add self.stretch_to_fit_depth(depth=mobject)
        else:
            self.scale_to_fit_dim(mobject, dim=dim_to_match)
        return self.set_position(position=mobject)

    def surround(
        self,
        mobject: Positionable,
        # TODO: Rename to `dim`
        dim_to_match: int = 0,
        stretch: bool = False,
        *,
        buff: float = MED_SMALL_BUFF,
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
        self.replace(mobject=mobject, dim_to_match=dim_to_match, stretch=stretch)
        size = mobject.get_dim_size(dim=dim_to_match)
        if size == 0:
            return self
        return self.scale((size + buff) / size)

    #############################
    ########## ALIASES ##########
    #############################

    # TODO: Only allow passing a single vector
    def shift(self, *vectors: Vector3DLike) -> Self:
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
        return self.translate(vector=vector)

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
        """
        return self.get_position(direction=direction)

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
        return self.get_position(direction=direction)

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
        return self.get_position(direction=direction)

    def move_to(
        self,
        point_or_mobject: Point3DLike | Positionable,
        aligned_edge: Vector3DLike = ORIGIN,
        # coor_mask: Vector3DLike = np.array([1, 1, 1]),
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
        return self.set_position(
            position=point_or_mobject,
            direction=aligned_edge,
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
        return self.set_coordinate(coordinate=value, dim=dim, direction=direction)

    def to_corner(
        self,
        corner: Vector3DLike = DL,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Moves to a corner.

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
        """
        return self.align_on_border(direction=corner, buff=buff)

    def to_edge(
        self,
        edge: Vector3DLike = LEFT,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        """Moves to an edge.

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
        """
        return self.align_on_border(direction=edge, buff=buff)

    @property
    def width(self) -> float:
        """The width.

        A property for the :meth:`get_width` and :meth:`set_width` methods.
        """
        return self.get_width()

    @width.setter
    def width(self, value: float) -> None:
        self.set_width(width=value)

    @property
    def height(self) -> float:
        """The height.

        A property for the :meth:`get_height` and :meth:`set_height` methods.
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
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.apply_points_function(
            func=func,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="set_dim_size")
    def rescale_to_fit(
        self,
        length: float | Positionable,
        dim: int,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=length,
            dim=dim,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="stretch")
    def stretch_about_point(
        self,
        factor: float,
        dim: int,
        point: Point3DLike,
    ) -> Self:
        return self.stretch(
            factor=factor,
            dim=dim,
            about_point=point,
        )

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

    # @deprecated(replacement="set_dim_size")
    def match_dim_size(
        self,
        mobject: Positionable,
        dim: int,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=mobject,
            dim=dim,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="set_width")
    def match_width(
        self,
        mobject: Positionable,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_width(
            width=mobject,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="set_height")
    def match_height(
        self,
        mobject: Positionable,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_height(
            height=mobject,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="set_depth")
    def match_depth(
        self,
        mobject: Positionable,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_depth(
            depth=mobject,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="set_coordinate")
    def match_coord(
        self,
        mobject: Positionable | float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_coordinate(
            coordinate=mobject,
            dim=dim,
            direction=direction,
        )

    # @deprecated(replacement="set_x")
    def match_x(self, mobject: Positionable, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_x(x=mobject, direction=direction)

    # @deprecated(replacement="set_y")
    def match_y(self, mobject: Positionable, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_y(y=mobject, direction=direction)

    # @deprecated(replacement="set_z")
    def match_z(self, mobject: Positionable, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_z(z=mobject, direction=direction)

    # @deprecated(replacement="set_points")
    def match_points(self, mobject: Positionable) -> Self:
        return self.set_points(points=mobject)

    # @deprecated(replacement="rotate")
    def rotate_about_origin(self, angle: float, axis: Vector3DLike = OUT) -> Self:
        return self.rotate(angle=angle, axis=axis, about_point=ORIGIN)

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
