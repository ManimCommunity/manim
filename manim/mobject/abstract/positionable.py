from collections.abc import Callable, Iterable
from typing import Any, Self

import numpy as np

from manim.constants import *
from manim.typing import *

Mobject = Any


class Positionable:
    __slots__ = ()

    def align_on_border(
        self,
        direction: Vector3DLike,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        raise NotImplementedError

    def align_to(
        self,
        mobject_or_point: Mobject | Point3DLike,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        raise NotImplementedError

    def apply_complex_function(
        self,
        function: Callable[[complex], complex],
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError

    def apply_function(
        self,
        function: MappingFunction,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError

    def apply_function_to_position(self, function: MappingFunction) -> Self:
        raise NotImplementedError

    def apply_matrix(
        self,
        matrix: MatrixMN,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError

    def apply_over_attr_arrays(self, func: MultiMappingFunction) -> Self:
        raise NotImplementedError

    def apply_points_function(
        self,
        func: MultiMappingFunction,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        works_on_bounding_box: bool = False,
    ) -> Self:
        raise NotImplementedError

    def apply_to_family(self, func: Callable[[Mobject], None]) -> Self:
        raise NotImplementedError

    def center(self) -> Self:
        raise NotImplementedError

    @property
    def depth(self) -> float:
        raise NotImplementedError

    @depth.setter
    def depth(self, value: float) -> None:
        raise NotImplementedError

    def flip(
        self,
        axis: Vector3DLike = UP,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError

    def get_bottom(self) -> Point3D:
        raise NotImplementedError

    def get_boundary_point(self, direction: Vector3DLike) -> Point3D:
        raise NotImplementedError

    def get_bounding_box(self) -> Point3D_Array:
        raise NotImplementedError

    def get_center(self) -> Point3D:
        raise NotImplementedError

    def get_center_of_mass(self) -> Point3D:
        raise NotImplementedError

    def get_coord(self, dim: int, direction: Vector3DLike = ORIGIN) -> float:
        raise NotImplementedError

    def get_corner(self, direction: Vector3DLike) -> Point3D:
        raise NotImplementedError

    def get_critical_point(self, direction: Vector3DLike) -> Point3D:
        raise NotImplementedError

    def get_depth(self) -> float:
        raise NotImplementedError

    def get_edge_center(self, direction: Vector3DLike) -> Point3D:
        raise NotImplementedError

    def get_extremum_along_dim(
        self,
        points: Point3DLike_Array | None = None,
        dim: int = 0,
        key: int = 0,
    ) -> float:
        raise NotImplementedError

    def get_height(self) -> float:
        raise NotImplementedError

    def get_left(self) -> Point3D:
        raise NotImplementedError

    def get_merged_array(self, array_attr: str) -> np.ndarray:
        raise NotImplementedError

    def get_midpoint(self) -> Point3D:
        raise NotImplementedError

    def get_nadir(self) -> Point3D:
        raise NotImplementedError

    def get_right(self) -> Point3D:
        raise NotImplementedError

    def get_top(self) -> Point3D:
        raise NotImplementedError

    def get_width(self) -> float:
        raise NotImplementedError

    def get_x(self, direction: Vector3DLike = ORIGIN) -> float:
        raise NotImplementedError

    def get_y(self, direction: Vector3DLike = ORIGIN) -> float:
        raise NotImplementedError

    def get_z(self, direction: Vector3DLike = ORIGIN) -> float:
        raise NotImplementedError

    def get_zenith(self) -> Point3D:
        raise NotImplementedError

    @property
    def height(self) -> float:
        raise NotImplementedError

    @height.setter
    def height(self, value: float) -> None:
        raise NotImplementedError

    def is_off_screen(self) -> bool:
        raise NotImplementedError

    def is_point_touching(
        self,
        point: Point3DLike,
        buff: float = MED_SMALL_BUFF,
    ) -> bool:
        raise NotImplementedError

    def length_over_dim(self, dim: int) -> float:
        raise NotImplementedError

    def match_coord(
        self,
        mobject: Mobject,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        raise NotImplementedError

    def match_depth(self, mobject: Mobject, **kwargs: Any) -> Self:
        raise NotImplementedError

    def match_dim_size(self, mobject: Mobject, dim: int, **kwargs: Any) -> Self:
        raise NotImplementedError

    def match_height(self, mobject: Mobject, **kwargs: Any) -> Self:
        raise NotImplementedError

    def match_width(self, mobject: Mobject, **kwargs: Any) -> Self:
        raise NotImplementedError

    def match_x(self, mobject: Mobject, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def match_y(self, mobject: Mobject, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def match_z(self, mobject: Mobject, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def move_to(
        self,
        point_or_mobject: Point3DLike | Mobject,
        aligned_edge: Vector3DLike = ORIGIN,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        raise NotImplementedError

    def next_to(
        self,
        mobject_or_point: Mobject | Point3DLike,
        direction: Vector3DLike = RIGHT,
        buff: float = DEFAULT_MOBJECT_TO_MOBJECT_BUFFER,
        aligned_edge: Vector3DLike = ORIGIN,
        submobject_to_align: Mobject | None = None,
        index_of_submobject_to_align: int | None = None,
        coor_mask: Vector3DLike = np.array([1, 1, 1]),
    ) -> Self:
        raise NotImplementedError

    def point_from_proportion(self, alpha: float) -> Point3D:
        raise NotImplementedError

    def pose_at_angle(self, **kwargs: Any) -> Self:
        raise NotImplementedError

    def proportion_from_point(self, point: Point3DLike) -> float:
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
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def rotate(
        self,
        angle: float,
        axis: Vector3DLike = OUT,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def rotate_about_origin(self, angle: float, axis: Vector3DLike = OUT) -> Self:
        raise NotImplementedError

    def scale(
        self,
        scale_factor: float,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def scale_to_fit_depth(
        self,
        depth: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def scale_to_fit_height(
        self,
        height: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def scale_to_fit_width(
        self,
        width: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def set_coord(
        self,
        value: float,
        dim: int,
        direction: Vector3DLike = ORIGIN,
    ) -> Self:
        raise NotImplementedError

    def set_depth(
        self,
        depth: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def set_height(
        self,
        height: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def set_width(
        self,
        width: float,
        stretch: bool = False,
        **kwargs: Any,
    ) -> Self:
        raise NotImplementedError

    def set_x(self, x: float, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def set_y(self, y: float, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def set_z(self, z: float, direction: Vector3DLike = ORIGIN) -> Self:
        raise NotImplementedError

    def shift(self, *vectors: Vector3DLike) -> Self:
        raise NotImplementedError

    def shift_onto_screen(self, **kwargs: Any) -> Self:
        raise NotImplementedError

    def stretch(
        self,
        factor: float,
        dim: int,
        *,
        about_point: Point3DLike | None = None,
        about_edge: Vector3DLike | None = None,
    ) -> Self:
        raise NotImplementedError

    def stretch_about_point(self, factor: float, dim: int, point: Point3DLike) -> Self:
        raise NotImplementedError

    def stretch_to_fit_depth(self, depth: float, **kwargs: Any) -> Self:
        raise NotImplementedError

    def stretch_to_fit_height(self, height: float, **kwargs: Any) -> Self:
        raise NotImplementedError

    def stretch_to_fit_width(self, width: float, **kwargs: Any) -> Self:
        raise NotImplementedError

    def to_corner(
        self,
        corner: Vector3DLike = DL,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        raise NotImplementedError

    def to_edge(
        self,
        edge: Vector3DLike = LEFT,
        buff: float = DEFAULT_MOBJECT_TO_EDGE_BUFFER,
    ) -> Self:
        raise NotImplementedError

    @property
    def width(self) -> float:
        raise NotImplementedError

    @width.setter
    def width(self, value: float) -> None:
        raise NotImplementedError
