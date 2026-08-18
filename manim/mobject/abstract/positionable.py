from collections.abc import Callable
from typing import Self

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
from manim.mobject.opengl.opengl_mobject import OpenGLMobject
from manim.mobject.opengl.opengl_vectorized_mobject import OpenGLVMobject
from manim.mobject.types.vectorized_mobject import VMobject
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

    # TODO: Keep/Remove frame parameter?
    # TODO: Should the default of the frame parameter be handled inside the method to allow config changes?
    def align_on_border(
        self,
        direction: Vector3DLike,
        *,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
        frame: Point3DLike | None = (config.frame_x_radius, config.frame_y_radius, 0),
    ) -> Self:
        target = np.sign(direction) * frame - buff * np.asarray(direction)
        return self.move_to(point_or_mobject=target, aligned_edge=direction)

    def align_to(
        self,
        mobject_or_point: "Positionable | Point3DLike",
        *,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        target = (
            mobject_or_point.get_critical_point(direction=direction)
            if isinstance(mobject_or_point, Positionable)
            else mobject_or_point
        )
        return self.move_to(point_or_mobject=target, aligned_edge=direction)

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

    def center(self) -> Self:
        return self.move_to(point_or_mobject=ORIGIN)

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
        return self.get_critical_point(direction=direction)[dim]

    def get_corner(self, direction: Vector3DLike) -> Point3D:
        return self.get_critical_point(direction=direction)

    def get_critical_point(self, direction: Vector3DLike) -> Point3D:
        direction = np.sign(direction)
        _, mids, maxs = self.get_bounding_box()
        return mids + (maxs - mids) * direction

    def get_depth(self) -> float:
        return self.length_over_dim(dim=2)

    def get_edge_center(self, direction: Vector3DLike) -> Point3D:
        return self.get_critical_point(direction=direction)

    def get_end(self) -> Point3D:
        return self.points[-1]

    def get_height(self) -> float:
        return self.length_over_dim(dim=1)

    def get_left(self) -> Point3D:
        return self.get_critical_point(direction=LEFT)

    def get_nadir(self) -> Point3D:
        return self.get_critical_point(direction=IN)

    def get_right(self) -> Point3D:
        return self.get_critical_point(direction=RIGHT)

    def get_start(self) -> Point3D:
        return self.points[0]

    def get_start_and_end(self) -> tuple[Point3D, Point3D]:
        return self.get_start(), self.get_end()

    def get_top(self) -> Point3D:
        return self.get_critical_point(UP)

    def get_width(self) -> float:
        return self.length_over_dim(dim=0)

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=0, direction=direction)

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=1, direction=direction)

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        return self.get_coord(dim=2, direction=direction)

    def get_zenith(self) -> Point3D:
        return self.get_critical_point(direction=OUT)

    def length_over_dim(self, dim: int) -> float:
        values = self.points[:, dim]
        return values.max() - values.min()

    def move_to(
        self,
        point_or_mobject: "Point3DLike | Positionable",
        *,
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        source = self.get_critical_point(direction=aligned_edge)
        target = (
            point_or_mobject.get_critical_point(direction=aligned_edge)
            if isinstance(point_or_mobject, Positionable)
            else point_or_mobject
        )
        return self.shift(vector=(target - source) * coor_mask)

    def next_to(
        self,
        mobject_or_point: "Positionable | Point3DLike",
        *,
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
        actual_length = self.length_over_dim(dim=dim)
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
        *,
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

    def stretch_to_fit(
        self,
        length: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        actual_length = self.length_over_dim(dim=dim)
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

    # Deprecated


def dump_methods() -> None:
    seen: set[str] = set()

    for cls in [Mobject, VMobject, OpenGLMobject, OpenGLVMobject]:
        assert isinstance(cls, type)
        print(cls.__name__)
        for name in sorted(cls.__dict__):
            if name in seen:
                continue
            print(f"\t{name}")
        seen |= cls.__dict__.keys()


if __name__ == "__main__":
    dump_methods()
