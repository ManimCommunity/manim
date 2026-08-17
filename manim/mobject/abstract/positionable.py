from collections.abc import Callable, Iterable
from typing import Any, Self

import numpy as np

from manim._config import config
from manim.constants import (
    DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
    DL,
    DOWN,
    IN,
    LEFT,
    ORIGIN,
    OUT,
    RIGHT,
    TAU,
    UP,
)
from manim.mobject.mobject import Mobject
from manim.typing import (
    MatrixMN,
    Point3D,
    Point3D_Array,
    Point3DLike,
    Vector3DLike,
)
from manim.utils.space_ops import rotation_matrix


class Positionable:
    # FUNDAMENTALS
    points: Point3D_Array

    # METHODS

    # TODO: Add a parameter for the frame?
    def align_on_border(
        self,
        direction: Vector3DLike,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        frame = (config.frame_x_radius, config.frame_y_radius, 0)
        target = np.sign(direction) * frame - buff * np.asarray(direction)
        self.move_to(point_or_mobject=target, aligned_edge=direction)
        return self

    def align_to(
        self,
        mobject_or_point: "Positionable | Point3DLike",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        target = mobject_or_point.get_critical_point(direction=direction) if isinstance(mobject_or_point, Positionable) else mobject_or_point
        self.move_to(point_or_mobject=target, aligned_edge=direction)
        return self

    def apply_complex_function(
        self,
        function: Callable[[complex], complex],
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        def R3_func(point: Point3D) -> Point3D:
            x, y, z = point
            xy_complex = function(complex(x, y))
            return np.array([xy_complex.real, xy_complex.imag, z])

        return self.apply_function(
            function=R3_func,
            about_point=about_point,
            about_edge=about_edge,
        )

    def apply_function(
        self,
        function: Callable[[Point3D], Point3D],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if about_point is None and about_edge is None:
            about_point = ORIGIN

        def multi_mapping_function(points: Point3D_Array) -> Point3D_Array:
            return np.apply_along_axis(func1d=function, axis=1, arr=points)

        return self.apply_points_function_about_point(
            function=multi_mapping_function,
            about_point=about_point,
            about_edge=about_edge,
        )

    def apply_function_to_position(
        self,
        function: Callable[[Point3D], Point3D],
    ) -> Self:
        return self.move_to(function(self.get_center()))

    def apply_matrix(
        self,
        matrix: MatrixMN,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if about_point is None and about_edge is None:
            about_point = ORIGIN

        matrix = np.asarray(matrix)

        # Fast path for standard 3x3 matrices
        if matrix.shape == (3, 3):
            full_matrix = matrix
        else:
            full_matrix = np.identity(3)
            full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

        return self.apply_points_function(
            lambda points: points.dot(full_matrix.T),
            about_point=about_point,
            about_edge=about_edge,
        )

    def apply_points_function(
        self,
        function: Callable[[Point3D], Point3D],
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if about_point is None:
            about_point = self.get_critical_point(direction=about_edge if about_edge is not None else ORIGIN)
        self.points -= about_point
        self.points = function(self.points)
        self.points += about_point
        return self

    # @deprecated(message="Use apply_points_function() instead.")
    def apply_points_function_about_point(
        self,
        function: Callable[[Point3D], Point3D],
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.apply_points_function(
            function=function,
            about_point=about_point,
            about_edge=about_edge,
        )

    def center(self) -> Self:
        return self.move_to(point_or_mobject=ORIGIN)

    @property
    def depth(self) -> float:
        return self.length_over_dim(dim=2)

    def flip(
        self,
        axis: Vector3DLike = UP,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.rotate(
            angle=TAU / 2,
            axis=axis,
            about_point=about_point,
            about_edge=about_edge,
        )

    def get_bottom(self) -> Point3D:
        return self.get_critical_point(DOWN)

    # TODO: Should this function be dropped?
    def get_boundary_point(self, direction: Vector3DLike) -> Point3D:
        index = np.argmax(np.dot(self.points, direction))
        return self.points[index]

    def get_bounding_box(self) -> Point3D_Array:
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        mids = (mins + maxs) / 2
        return np.array([mins, mids, maxs])

    def get_center(self) -> Point3D:
        return self.get_critical_point(ORIGIN)

    def get_center_of_mass(self) -> Point3D:
        return self.points.mean(axis=0)

    def get_coord(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_critical_point(direction=direction)[dim]

    # @deprecated(message="Use get_critical_point() instead")
    def get_corner(self, direction: Vector3DLike) -> Point3D:
        return self.get_critical_point(direction)

    # TODO: Should the `np.sign(direction)` restriction be dropped?
    # Advantage: Would allow in-between values
    # Disadvantage: Would alter behavior
    # Alternative: Declare an additional method. (which this method then would use)
    def get_critical_point(self, direction: Vector3DLike) -> Point3D:
        direction = np.sign(direction)
        _, mids, maxs = self.get_bounding_box()
        return mids + (maxs - mids) * direction

    def get_edge_center(self, direction: Vector3DLike) -> Point3D:
        return self.get_critical_point(direction=direction)

    def get_end(self) -> Point3D:
        return self.points[-1]

    def get_extremum_along_dim(
        self,
        dim: int = 0,
        key: int = 0,
    ) -> float:
        direction = np.zeros(3)
        direction[dim] = np.sign(key)
        critical_pt = self.get_critical_point(direction)
        return critical_pt[dim]

    def get_left(self) -> Point3D:
        return self.get_critical_point(LEFT)

    def get_nadir(self) -> Point3D:
        """Get nadir (opposite the zenith) Point3Ds of a box bounding a 3D :class:`~.Mobject`."""
        return self.get_critical_point(IN)

    def get_right(self) -> Point3D:
        return self.get_critical_point(RIGHT)

    def get_start(self) -> Point3D:
        return self.points[0]

    def get_start_and_end(self) -> tuple[Point3D, Point3D]:
        return self.get_start(), self.get_end()

    def get_top(self) -> Point3D:
        return self.get_critical_point(UP)

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=0, direction=direction)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=1, direction=direction)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=2, direction=direction)

    def get_zenith(self) -> Point3D:
        return self.get_critical_point(direction=OUT)

    @property
    def height(self) -> float:
        return self.length_over_dim(dim=1)

    @height.setter
    def height(self, value: float) -> None:
        raise NotImplementedError

    def length_over_dim(self, dim: int) -> float:
        values = self.points[:, dim]
        return values.max() - values.min()

    def match_coord(
        self,
        mobject: Mobject,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_coord(
            mobject.get_coord(dim=dim, direction=direction),
            dim=dim,
            direction=direction,
        )

    # def match_depth(self) -> Self:
    #    return self.set_depth()

    # def match_dim_size(self) -> Self:
    #    return self.set_dim_size()

    # def match_height(self) -> Self:
    #    return self.set_height()

    def match_points(self, mobject: "Positionable") -> Self:
        self.points = mobject.points.copy()
        return self

    # def match_width(self) -> Self:
    #    return self.set_width()

    def match_x(
        self,
        mobject: "Positionable",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_x(
            x=mobject.get_x(direction=direction),
            direction=direction,
        )

    def match_y(
        self,
        mobject: "Positionable",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_y(
            y=mobject.get_y(direction=direction),
            direction=direction,
        )

    def match_z(
        self,
        mobject: "Positionable",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_z(
            z=mobject.get_z(direction=direction),
            direction=direction,
        )

    def move_to(
        self,
        point_or_mobject: "Point3DLike | Positionable",
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        source = self.get_critical_point(aligned_edge)
        target = point_or_mobject.get_critical_point(aligned_edge) if isinstance(point_or_mobject, Positionable) else point_or_mobject
        self.shift((target - source) * coor_mask)
        return self

    def next_to(
        self,
        mobject_or_point: "Positionable | Point3DLike",
        direction: Vector3DLike = RIGHT,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        np_direction = np.asarray(direction)
        np_aligned_edge = np.asarray(aligned_edge)
        source = self.get_critical_point(np_aligned_edge - np_direction)
        target = mobject_or_point.get_critical_point(np_aligned_edge + np_direction) if isinstance(mobject_or_point, Positionable) else mobject_or_point
        return self.shift((target - source + buff * np_direction) * coor_mask)

    def pose_at_angle(self, **kwargs: Any) -> Self:
        raise NotImplementedError

    def put_start_and_end_on(self, start: Point3DLike, end: Point3DLike) -> Self:
        raise NotImplementedError

    def reduce_across_dimension(
        self,
        reduce_func: Callable[[Iterable[float]], float],
        dim: int,
    ) -> float | None:
        raise NotImplementedError

    def rescale_to_fit(
        self,
        length: float,
        dim: int,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError
        # old_length = self.length_over_dim(dim=dim)
        # if old_length == 0:
        #    return self
        # if stretch:
        #    self.stretch(length / old_length, dim, ...)
        # else:
        #    self.scale(length / old_length, ...)
        # return self

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.apply_matrix(
            matrix=rotation_matrix(angle, axis),
            about_point=about_point,
            about_edge=about_edge,
        )

    def rotate_about_origin(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
    ) -> Self:
        return self.rotate(
            angle=angle,
            axis=axis,
            about_point=ORIGIN,
        )

    def scale(
        self,
        scale_factor: float,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError

    def scale_to_fit_depth(self) -> Self:
        raise NotImplementedError

    def scale_to_fit_height(self) -> Self:
        raise NotImplementedError

    def scale_to_fit_width(self) -> Self:
        raise NotImplementedError

    def set_coord(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        target = self.get_critical_point(direction=direction)
        target[dim] = value
        return self.move_to(point_or_mobject=target, aligned_edge=direction)

    def set_x(
        self,
        x: float,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_coord(value=x, dim=0, direction=direction)

    def set_y(self, y: float, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_coord(value=y, dim=1, direction=direction)

    def set_z(self, z: float, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_coord(value=z, dim=2, direction=direction)

    def shift(self, vector: Vector3DLike) -> Self:
        self.points += vector
        return self

    def shift_onto_screen(self) -> Self:
        raise NotImplementedError

    def stretch(
        self,
        factor: float,
        dim: int,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        def function(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        return self.apply_points_function(
            function=function,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_about_point(self) -> Self:
        raise NotImplementedError

    def stretch_to_fit_depth(self) -> Self:
        raise NotImplementedError

    def stretch_to_fit_height(self) -> Self:
        raise NotImplementedError

    def stretch_to_fit_width(self) -> Self:
        raise NotImplementedError

    def to_corner(
        self,
        corner: Vector3DLike = DL,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        return self.align_on_border(direction=corner, buff=buff)

    def to_edge(
        self,
        edge: Vector3DLike = LEFT,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        return self.align_on_border(direction=edge, buff=buff)

    @property
    def width(self) -> float:
        return self.length_over_dim(dim=0)
