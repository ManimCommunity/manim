from collections.abc import Callable, Iterable
from typing import Any, Self

import numpy as np

from manim._config import config
from manim.constants import (
    DEFAULT_MOBJECT_TO_EDGE_BUFFER,
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
    Point3DLike_Array,
    Vector3DLike,
)
from manim.utils.space_ops import rotation_matrix


class Positionable:
    """A positionable object.

    ### Basics
        - points
        - (get|set)_points
        -  get_points_defining_boundary
    ### Applying Functions
        - get_family
        - apply_to_family
        - apply_array_function
        - apply_function
        - apply_complex_function
    ### Transformations
        - apply_matrix
        - translate
        - rotate
        - scale
        - stretch
    ### General
        - get_bounding_box
        - (get|set)_position
            - (get|set)_(center|left|right|bottom|top|nadir|zenith)
        - (get|set)_coord
            - (get|set)_(x|y|z)
        - (get|set)_dim_size
            - (get|set)_(width|height|depth)
    ### Specialized
        - align_to
            - align_on_border
            - next_to (TODO)
        - is_off_screen
        - get_center_of_mass
        - get_boundary_point
        - shift_onto_screen
    ### Aliases
        - center = set_center(ORIGIN)
        - flip
        - length_over_dim
        - move_to = set_position
        - scale_to_fit = set_dim_size(stretch=False)
            - scale_to_fit_(width|height|depth)
        - stretch_to_fit = set_dim_size(stretch=True)
            - stretch_to_fit_(width|height|depth)
        - get_critical_point = get_position
        - get_edge_center = get_position
        - get_corner = get_position
        - pose_at_angle
        - shift = translate
        - to_corner = align_on_border
        - to_edge = align_on_border
        - (width|height|depth)
    ### Deprecated
        - apply_points_function_about_point
        - apply_function_to_position
        - get_extremum_along_dim
        - match_points
        - match_coord
        - match_(x|y|z)
        - match_dim_size
        - match_(width|height|depth)
        - reduce_across_dimension
        - rescale_to_fit
        - rotate_about_origin
        - stretch_about_point

    """

    ### FUNDAMENTALS ###
    points: Point3D_Array = np.array([])

    def get_points(self) -> Point3D_Array:
        return np.concat([mob.points for mob in self.get_family()])

    def set_points(
        self,
        points: "Point3DLike_Array | Positionable",
    ) -> Self:
        if isinstance(points, Positionable):
            for mob1, mob2 in zip(self.get_family(), points.get_family(), strict=False):
                mob1.set_points(mob2.points.copy())
        else:
            self.points = np.asarray(points)
        return self

    def get_points_defining_boundary(self) -> Point3D_Array:
        return self.get_points()

    ### APPLYING FUNCTIONS ###

    def get_family(self) -> Iterable["Positionable"]:
        yield self

    def apply_to_family(
        self,
        function: Callable[["Positionable"], Any],
        *,
        only_with_points: bool = True,
    ) -> Self:
        for mob in self.get_family():
            if only_with_points and len(mob.points) == 0:
                continue
            function(mob)
        return self

    def apply_array_function(
        self,
        function: Callable[[Point3D_Array], Point3D_Array],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if about_point is None:
            if about_edge is None:
                about_edge = ORIGIN
            about_point = self.get_position(direction=about_edge)

        about_point = np.array(about_point, copy=True)

        def apply(mob: Positionable) -> None:
            mob.points -= about_point
            mob.points = function(mob.points)
            mob.points += about_point

        return self.apply_to_family(function=apply)

    def apply_function(
        self,
        function: Callable[[Point3D], Point3D],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if about_point is None and about_edge is None:
            about_point = ORIGIN

        def apply(points: Point3D_Array) -> Point3D_Array:
            return np.apply_along_axis(func1d=function, axis=1, arr=points)

        return self.apply_array_function(
            function=apply,
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
        def apply(point: Point3D) -> Point3D:
            x, y, z = point
            xy_complex = function(complex(x, y))
            return np.array([xy_complex.real, xy_complex.imag, z])

        return self.apply_function(
            function=apply,
            about_point=about_point,
            about_edge=about_edge,
        )

    ### TRANSFORMATIONS ###

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
        full_matrix = np.identity(3)
        full_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix

        self.apply_array_function(
            function=lambda points: points.dot(full_matrix.T, out=points),
            about_point=about_point,
            about_edge=about_edge,
        )
        return self

    def translate(self, vector: Vector3DLike) -> Self:
        def function(mob: Positionable) -> None:
            mob.points += vector

        return self.apply_to_family(function=function)

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

    def scale(
        self,
        # TODO: Rename to `factor`
        scale_factor: float,
        scale_stroke: bool = False,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.apply_array_function(
            function=lambda points: points.__imul__(scale_factor),
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
        def function(points: Point3D_Array) -> Point3D_Array:
            points[:, dim] *= factor
            return points

        return self.apply_array_function(
            function=function,
            about_point=about_point,
            about_edge=about_edge,
        )

    ### GENERAL ###

    def get_bounding_box(self) -> tuple[Point3D, Point3D]:
        points = self.get_points_defining_boundary()
        if len(points) == 0:
            return (np.zeros(3), np.zeros(3))
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return (mins, maxs)

    def get_position(
        self,
        direction: Vector3DLike = ORIGIN,
    ) -> Point3D:
        direction = np.sign(direction)
        mins, maxs = self.get_bounding_box()
        mids = (mins + maxs) / 2
        return mids + (maxs - mids) * direction

    def set_position(
        self,
        point: "Point3DLike | Positionable",
        *,
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        if isinstance(point, Positionable):
            point = point.get_position(direction=aligned_edge)
        current = self.get_position(direction=aligned_edge)
        vector = (point - current) * coor_mask
        return self.translate(vector=vector)

    def get_center(self) -> Point3D:
        return self.get_position(direction=ORIGIN)

    def set_center(
        self,
        center: "Point3DLike | Positionable",
        *,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(point=center, aligned_edge=ORIGIN, coor_mask=coor_mask)

    def get_left(self) -> Point3D:
        return self.get_position(direction=LEFT)

    def set_left(
        self,
        left: "Point3DLike | Positionable",
        *,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(point=left, aligned_edge=LEFT, coor_mask=coor_mask)

    def get_right(self) -> Point3D:
        return self.get_position(direction=RIGHT)

    def set_right(
        self,
        right: "Point3DLike | Positionable",
        *,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(point=right, aligned_edge=RIGHT, coor_mask=coor_mask)

    def get_bottom(self) -> Point3D:
        return self.get_position(direction=DOWN)

    def set_bottom(
        self,
        bottom: "Point3DLike | Positionable",
        *,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(point=bottom, aligned_edge=DOWN, coor_mask=coor_mask)

    def get_top(self) -> Point3D:
        return self.get_position(direction=UP)

    def set_top(
        self,
        top: "Point3DLike | Positionable",
        *,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(point=top, aligned_edge=UP, coor_mask=coor_mask)

    def get_nadir(self) -> Point3D:
        return self.get_position(direction=IN)

    def set_nadir(
        self,
        nadir: "Point3DLike | Positionable",
        *,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(point=nadir, aligned_edge=IN, coor_mask=coor_mask)

    def get_zenith(self) -> Point3D:
        return self.get_position(direction=OUT)

    def set_zenith(
        self,
        zenith: "Point3DLike | Positionable",
        *,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(point=zenith, aligned_edge=OUT, coor_mask=coor_mask)

    def get_coord(
        self,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> float:
        points = self.get_points()
        if len(points) == 0:
            return 0

        key = direction[dim]
        values = points[:, dim]
        return (  # type: ignore[no-any-return]
            values.min()
            if key < 0
            else (values.min() + values.max()) / 2
            if key == 0
            else values.max()
        )

    def set_coord(
        self,
        value: "float | Positionable",
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        if isinstance(value, Positionable):
            value = value.get_coord(dim=dim, direction=direction)
        current = self.get_coord(dim=dim, direction=direction)
        vector = np.zeros(3)
        vector[dim] = value - current
        return self.translate(vector=vector)

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=0, direction=direction)

    def set_x(self, x: float, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_coord(value=x, dim=0, direction=direction)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=1, direction=direction)

    def set_y(self, y: float, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_coord(value=y, dim=1, direction=direction)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=2, direction=direction)

    def set_z(self, z: float, direction: Vector3DLike = ORIGIN) -> Self:
        return self.set_coord(value=z, dim=2, direction=direction)

    def get_dim_size(self, dim: int) -> float:
        points = self.get_points()
        if len(points) == 0:
            return 0
        return np.ptp(points[:, dim])  # type: ignore[no-any-return]

    def set_dim_size(
        self,
        size: "float | Positionable",
        dim: int,
        *,
        stretch: bool = False,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        if isinstance(size, Positionable):
            size = size.get_dim_size(dim=dim)

        current_size = self.get_dim_size(dim=dim)
        if current_size == 0:
            return self

        factor = size / current_size
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
        return self.get_dim_size(dim=0)

    def set_width(
        self,
        width: "float | Positionable",
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

    def get_height(self) -> float:
        return self.get_dim_size(dim=1)

    def set_height(
        self,
        height: "float | Positionable",
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

    def get_depth(self) -> float:
        return self.get_dim_size(dim=2)

    def set_depth(
        self,
        depth: "float | Positionable",
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

    ### SPECIALIZED ###

    def align_on_border(
        self,
        direction: Vector3DLike,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        frame = (config.frame_x_radius, config.frame_y_radius, 0)
        target = np.sign(direction) * frame - buff * np.asarray(direction)
        return self.align_to(target, direction=direction)

    def align_to(
        self,
        # TODO: Rename to point
        mobject_or_point: "Positionable | Point3DLike",
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        if isinstance(mobject_or_point, Positionable):
            mobject_or_point = mobject_or_point.get_position(direction=direction)
        source = self.get_critical_point(direction=direction)
        target = np.where(direction == 0, source, mobject_or_point)
        return self.shift(target - source)

    def is_off_screen(self) -> bool:
        mins, maxs = self.get_bounding_box()
        return (  # type: ignore[return-value]
            mins[0] > config.frame_x_radius
            or maxs[0] < -config.frame_x_radius
            or mins[1] > config.frame_y_radius
            or maxs[1] < -config.frame_y_radius,
        )

    def get_center_of_mass(self) -> Point3D:
        points = self.get_points()
        if len(points) == 0:
            return ORIGIN
        return points.mean(axis=0)

    def get_boundary_point(self, direction: Vector3DLike) -> Point3D:
        points = self.get_points_defining_boundary()
        index = np.argmax(points.dot(direction))
        return points[index]

    def shift_onto_screen(
        self,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        # TODO: Simplify implementation
        space_lengths = [config.frame_x_radius, config.frame_y_radius]
        for vect in UP, DOWN, LEFT, RIGHT:
            dim = np.argmax(np.abs(vect))
            max_val = space_lengths[dim] - buff
            edge_center = self.get_edge_center(vect)
            if np.dot(edge_center, vect) > max_val:
                self.to_edge(vect, buff=buff)
        return self

    ### ALIASES ###
    shift = translate
    get_critical_point = get_position
    get_edge_center = get_position
    get_corner = get_position
    length_over_dim = get_dim_size

    def center(self) -> Self:
        return self.set_center(ORIGIN)

    def flip(
        self,
        axis: Vector3DLike = UP,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.rotate(
            TAU / 2,
            axis,
            about_point=about_point,
            about_edge=about_edge,
        )

    def move_to(
        self,
        point_or_mobject: "Point3DLike | Positionable",
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        return self.set_position(
            point=point_or_mobject,
            aligned_edge=aligned_edge,
            coor_mask=coor_mask,
        )

    def pose_at_angle(
        self,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.rotate(
            angle=TAU / 14,
            axis=RIGHT + UP,
            about_point=about_point,
            about_edge=about_edge,
        )

    def scale_to_fit(
        self,
        size: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=size,
            dim=dim,
            stretch=False,
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
            size=width,
            dim=0,
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
            size=height,
            dim=1,
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
            size=depth,
            dim=2,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit(
        self,
        size: float,
        dim: int,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.set_dim_size(
            size=size,
            dim=dim,
            stretch=True,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_width(
        self,
        width: float,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.stretch_to_fit(
            size=width,
            dim=0,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_height(
        self,
        height: float,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.stretch_to_fit(
            size=height,
            dim=1,
            about_point=about_point,
            about_edge=about_edge,
        )

    def stretch_to_fit_depth(
        self,
        depth: float,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        return self.stretch_to_fit(
            size=depth,
            dim=2,
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
    def width(self) -> float:
        return self.get_width()

    @width.setter
    def width(self, value: float) -> None:
        self.set_width(width=value)

    @property
    def height(self) -> float:
        return self.get_height()

    @height.setter
    def height(self, value: float) -> None:
        self.set_height(height=value)

    @property
    def depth(self) -> float:
        return self.get_depth()

    @depth.setter
    def depth(self, value: float) -> None:
        self.set_depth(depth=value)

    ### DEPRECATED ###

    apply_points_function_about_point = apply_array_function
    match_points = set_points
    match_coord = set_coord
    match_x = set_x
    match_y = set_y
    match_z = set_z
    match_dim_size = set_dim_size
    match_width = set_width
    match_height = set_height
    match_depth = set_depth

    # @deprecated(replacement="move_to(function(self.get_center()))")
    def apply_function_to_position(
        self,
        function: Callable[[Point3D], Point3DLike],
    ) -> Self:
        return self.move_to(function(self.get_center()))

    # @deprecated()
    def get_extremum_along_dim(
        self,
        dim: int = 0,
        key: int = 0,
    ) -> float:
        points = self.get_points()
        if len(points) == 0:
            return 0
        values = points[:, dim]
        if key < 0:
            rv: float = np.min(values)
            return rv
        elif key == 0:
            rv = (np.min(values) + np.max(values)) / 2
            return rv
        else:
            rv = np.max(values)
            return rv

    # @deprecated()
    def reduce_across_dimension(
        self,
        reduce_func: Callable[[Iterable[float]], float],
        dim: int,
    ) -> float | None:
        points = self.get_points()
        if len(points) == 0:
            return None

        return reduce_func(points[:, dim])

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

    # @deprecated(replacement="set_dim_size")
    def rescale_to_fit(
        self,
        length: "float | Positionable",
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
    def stretch_about_point(self, factor: float, dim: int, point: Point3DLike) -> Self:
        return self.stretch(factor=factor, dim=dim, about_point=point)
