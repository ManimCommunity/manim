"""Utility functions for three-dimensional mobjects."""

from __future__ import annotations

__all__ = [
    "get_3d_vmob_gradient_start_and_end_points",
    "get_3d_vmob_start_corner_index",
    "get_3d_vmob_end_corner_index",
    "get_3d_vmob_start_corner",
    "get_3d_vmob_end_corner",
    "get_3d_vmob_unit_normal",
    "get_3d_vmob_start_corner_unit_normal",
    "get_3d_vmob_end_corner_unit_normal",
]


from typing import TYPE_CHECKING, Literal

from manim.utils._path_points import (
    end_corner,
    end_corner_index,
    start_corner,
    unit_normal,
)

if TYPE_CHECKING:
    from manim.typing import Point3D, Vector3D

    from ..types.vectorized_mobject import VMobject


def get_3d_vmob_gradient_start_and_end_points(
    vmob: VMobject,
) -> tuple[Point3D, Point3D]:
    return (
        get_3d_vmob_start_corner(vmob),
        get_3d_vmob_end_corner(vmob),
    )


def get_3d_vmob_start_corner_index(vmob: VMobject) -> Literal[0]:
    return 0


def get_3d_vmob_end_corner_index(vmob: VMobject) -> int:
    return end_corner_index(vmob.points)


def get_3d_vmob_start_corner(vmob: VMobject) -> Point3D:
    return start_corner(vmob.points)


def get_3d_vmob_end_corner(vmob: VMobject) -> Point3D:
    return end_corner(vmob.points)


def get_3d_vmob_unit_normal(vmob: VMobject, point_index: int) -> Vector3D:
    return unit_normal(vmob.points, point_index)


def get_3d_vmob_start_corner_unit_normal(vmob: VMobject) -> Vector3D:
    return unit_normal(vmob.points, 0)


def get_3d_vmob_end_corner_unit_normal(vmob: VMobject) -> Vector3D:
    return unit_normal(vmob.points, end_corner_index(vmob.points))
