from collections.abc import Callable, Iterable
from typing import Self
from warnings import deprecated

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
from manim.typing import (
    MatrixMN,
    Point3D,
    Point3D_Array,
    Point3DLike,
    Vector3DLike,
)

# from manim.utils.deprecation import deprecated
from manim.utils.space_ops import rotation_matrix


class Positionable:
    points: Point3D_Array = np.array([(0.0, 0.0, 0.0)])

    def align_on_border(
        self,
        direction: Vector3DLike,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        *,
        frame: Point3DLike | None = None,
    ) -> Self:
        if frame is None:
            frame = (config.frame_x_radius, config.frame_y_radius, 0)
        target_point = np.sign(direction) * frame
        point_to_align = self.get_critical_point(direction=direction)
        shift_val = target_point - point_to_align - buff * np.asarray(direction)
        shift_val = shift_val * abs(np.sign(direction))
        return self.shift(shift_val)

    def align_to(
        self,
        mobject_or_point: "Positionable | Point3DLike",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        source = self.get_critical_point(direction=direction)
        target = np.array(
            mobject_or_point.get_critical_point(direction=direction)
            if isinstance(mobject_or_point, Positionable)
            else mobject_or_point
        )
        target = np.where(direction == 0, source, target)
        return self.shift(target - source)

    def apply_array_function(
        self,
        function: Callable[[Point3D_Array], Point3D_Array],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if about_point is None:
            about_point = self.get_critical_point(
                direction=about_edge if about_edge is not None else ORIGIN
            )
        self.points -= about_point
        self.points = function(self.points)
        self.points += about_point
        return self

    def apply_complex_function(
        self,
        function: Callable[[complex], complex],
        *,
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

        def mapping_function(points: Point3D_Array) -> Point3D_Array:
            return np.apply_along_axis(func1d=function, axis=1, arr=points)

        return self.apply_array_function(
            function=mapping_function,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="move_to(function(self.get_center()))")
    def apply_function_to_position(
        self,
        function: Callable[[Point3D], Point3D],
    ) -> Self:
        return self.move_to(function(self.get_center()))

    def apply_matrix(
        self,
        matrix: MatrixMN,
        *,
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

        return self.apply_array_function(
            function=lambda points: points.dot(full_matrix.T),
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="apply_array_function")
    def apply_points_function_about_point(
        self,
        func: Callable[[Point3D_Array], Point3D_Array],
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.apply_array_function(
            function=func,
            about_point=about_point,
            about_edge=about_edge,
        )

    def center(self) -> Self:
        return self.move_to(point_or_mobject=ORIGIN)

    @property
    # @deprecated(replacement="get_depth")
    def depth(self) -> float:
        return self.get_depth()

    @depth.setter
    # @deprecated(replacement="set_depth")
    def depth(self, value: float) -> Self:
        return self.set_depth(depth=value, stretch=False)

    def flip(
        self,
        axis: Vector3DLike = UP,
        *,
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
        return self.get_critical_point(direction=DOWN)

    def get_boundary_point(self, direction: Vector3DLike) -> Point3D:
        index = np.argmax(np.dot(self.points, direction))
        return self.points[index]

    def get_bounding_box(self) -> Point3D_Array:
        mins = self.points.min(axis=0)
        maxs = self.points.max(axis=0)
        mids = (mins + maxs) / 2
        return np.array([mins, mids, maxs])

    def get_center(self) -> Point3D:
        return self.get_critical_point(direction=ORIGIN)

    def get_center_of_mass(self) -> Point3D:
        return self.points.mean(axis=0)

    def get_coord(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        key = np.sign(direction[dim])
        return (
            self.points[:, dim].min()
            if key == -1
            else (self.points[:, dim].min() + self.points[:, dim].max()) / 2
            if key == 0
            else self.points[:, dim].max()
        )

    # @deprecated(replacement="get_critical_point")
    def get_corner(self, direction: Vector3DLike) -> Point3D:
        return self.get_critical_point(direction=direction)

    def get_critical_point(self, direction: Vector3DLike) -> Point3D:
        direction = np.sign(direction)
        _, mids, maxs = self.get_bounding_box()
        return mids + (maxs - mids) * direction

    def get_depth(self) -> float:
        return self.get_dim_size(dim=2)

    def get_dim_size(self, dim: int) -> float:
        values = self.points[:, dim]
        return values.max() - values.min()

    # @deprecated(replacement="get_critical_point")
    def get_edge_center(self, direction: Vector3DLike) -> Point3D:
        return self.get_critical_point(direction=direction)

    # @deprecated()
    def get_extremum_along_dim(
        self,
        dim: int = 0,
        key: int = 0,
    ) -> float:
        values = self.points[:, dim]
        if key < 0:
            rv: float = np.min(values)
            return rv
        elif key == 0:
            rv = (np.min(values) + np.max(values)) / 2
            return rv
        else:
            rv = np.max(values)
            return rv

    def get_height(self) -> float:
        return self.get_dim_size(dim=1)

    def get_left(self) -> Point3D:
        return self.get_critical_point(direction=LEFT)

    def get_nadir(self) -> Point3D:
        return self.get_critical_point(direction=IN)

    def get_right(self) -> Point3D:
        return self.get_critical_point(direction=RIGHT)

    def get_top(self) -> Point3D:
        return self.get_critical_point(UP)

    def get_width(self) -> float:
        return self.get_dim_size(dim=0)

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=0, direction=direction)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=1, direction=direction)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=2, direction=direction)

    def get_zenith(self) -> Point3D:
        return self.get_critical_point(direction=OUT)

    @property
    # @deprecated(replacement="get_height")
    def height(self) -> float:
        return self.get_height()

    @height.setter
    # @deprecated(replacement="set_height")
    def height(self, value: float) -> Self:
        return self.set_height(height=value, stretch=False)

    def is_off_screen(self) -> bool:
        mins, _, maxs = self.get_bounding_box()
        return (
            mins[0] > config.frame_x_radius
            or maxs[0] < -config.frame_x_radius
            or mins[1] > config.frame_y_radius
            or maxs[1] < -config.frame_y_radius
        )

    # @deprecated(replacement="get_dim_size")
    def length_over_dim(self, dim: int) -> float:
        return self.get_dim_size(dim=dim)

    def match_coord(
        self,
        mobject: "Positionable",
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_coord(
            mobject.get_coord(dim=dim, direction=direction),
            dim=dim,
            direction=direction,
        )

    def match_depth(
        self,
        mobject: "Positionable",
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_depth(
            mobject.get_depth(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def match_dim_size(
        self,
        mobject: "Positionable",
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            mobject.get_dim_size(dim=dim),
            dim=dim,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def match_height(
        self,
        mobject: "Positionable",
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_height(
            mobject.get_height(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def match_points(self, mobject: "Positionable") -> Self:
        self.points = mobject.points.copy()
        return self

    def match_width(
        self,
        mobject: "Positionable",
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_width(
            mobject.get_width(),
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def match_x(
        self,
        mobject: "Positionable",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_x(
            mobject.get_x(direction=direction),
            direction=direction,
        )

    def match_y(
        self,
        mobject: "Positionable",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_y(
            mobject.get_y(direction=direction),
            direction=direction,
        )

    def match_z(
        self,
        mobject: "Positionable",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_z(
            mobject.get_z(direction=direction),
            direction=direction,
        )

    def move_to(
        self,
        point_or_mobject: "Point3DLike | Positionable",
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        if isinstance(point_or_mobject, Positionable):
            return self.move_to(
                point_or_mobject.get_critical_point(direction=aligned_edge),
                aligned_edge=aligned_edge,
                coor_mask=coor_mask,
            )
        source = self.get_critical_point(direction=aligned_edge)
        target = point_or_mobject
        return self.shift(vector=(target - source) * coor_mask)

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
        source = self.get_critical_point(direction=np_aligned_edge - np_direction)
        target = (
            mobject_or_point.get_critical_point(np_aligned_edge + np_direction)
            if isinstance(mobject_or_point, Positionable)
            else mobject_or_point
        )
        return self.shift((target - source + buff * np_direction) * coor_mask)

    # @deprecated()
    def pose_at_angle(
        self,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.rotate(
            angle=TAU / 14,
            axis=RIGHT + UP,
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated()
    def reduce_across_dimension(
        self,
        reduce_func: Callable[[Iterable[float]], float],
        dim: int,
    ) -> float | None:
        if len(self.points) == 0:
            return None

        return reduce_func(self.points[:, dim])

    # @deprecated(replacement="set_dim_size")
    def rescale_to_fit(
        self,
        length: float,
        dim: int,
        stretch: bool = False,
        *,
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

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if about_point is None and about_edge is None:
            about_edge = ORIGIN
        return self.apply_matrix(
            matrix=rotation_matrix(angle, axis),
            about_point=about_point,
            about_edge=about_edge,
        )

    # @deprecated(replacement="rotate")
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
        # TODO: Rename to 'factor'
        scale_factor: float | Vector3DLike,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.apply_array_function(
            function=lambda points: scale_factor * points,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit(
        self,
        length: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        actual_length = self.get_dim_size(dim=dim)
        if actual_length == 0:
            return self

        return self.scale(
            scale_factor=length / actual_length,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit_depth(
        self,
        depth: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.scale_to_fit(
            length=depth,
            dim=2,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit_height(
        self,
        height: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.scale_to_fit(
            length=height,
            dim=1,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit_width(
        self,
        width: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.scale_to_fit(
            length=width,
            dim=0,
            about_point=about_point,
            about_edge=about_edge,
        )

    def set_coord(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        source = self.get_coord(dim, direction=direction)
        self.points[:, dim] += value - source
        return self

    def set_depth(
        self,
        depth: float,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=depth,
            dim=2,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def set_dim_size(
        self,
        size: float,
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        old_length = self.get_dim_size(dim=dim)
        if old_length == 0:
            return self
        factor = size / old_length
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

    def set_height(
        self,
        height: float,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=height,
            dim=1,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def set_width(
        self,
        width: float,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=width,
            dim=0,
            stretch=stretch,
            about_point=about_point,
            about_edge=about_edge,
        )

    def set_x(
        self,
        x: float,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_coord(value=x, dim=0, direction=direction)

    def set_y(
        self,
        y: float,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_coord(value=y, dim=1, direction=direction)

    def set_z(
        self,
        z: float,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        return self.set_coord(value=z, dim=2, direction=direction)

    def shift(self, vector: Vector3DLike) -> Self:
        self.points += vector
        return self

    # @deprecated()
    def shift_onto_screen(
        self,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        # TODO: Simplify implementation
        space_lengths = [config["frame_x_radius"], config["frame_y_radius"]]
        for edge in UP, DOWN, LEFT, RIGHT:
            dim = np.argmax(np.abs(edge))
            max_val = space_lengths[dim] - buff
            edge_center = self.get_critical_point(direction=edge)
            if np.dot(edge_center, edge) > max_val:
                self.to_edge(edge=edge, buff=buff)
        return self

    def stretch(
        self,
        factor: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.scale(
            scale_factor=np.array([factor if i == dim else 1.0 for i in range(3)]),
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

    def stretch_to_fit(
        self,
        length: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        actual_length = self.get_dim_size(dim=dim)
        if actual_length == 0:
            return self

        return self.stretch(
            factor=length / actual_length,
            dim=dim,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_depth(
        self,
        depth: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.stretch_to_fit(
            length=depth,
            dim=2,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_height(
        self,
        height: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.stretch_to_fit(
            length=height,
            dim=1,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_width(
        self,
        width: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.stretch_to_fit(
            length=width,
            dim=0,
            about_point=about_point,
            about_edge=about_edge,
        )

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
    # @deprecated(replacement="get_width")
    def width(self) -> float:
        return self.get_width()

    @width.setter
    def width(self, value: float) -> Self:
        return self.set_width(width=value, stretch=False)
