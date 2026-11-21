"""Functions on the points of a cubic Bézier path, without a mobject.

Renderers draw paths from render snapshots, which carry the points and tolerance of a
path but not the :class:`~.VMobject`; these functions need only those. The VMobject
methods of the same purpose delegate to them.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from manim.constants import ORIGIN, UP
from manim.utils.space_ops import get_unit_normal

if TYPE_CHECKING:
    import numpy.typing as npt

    from manim.typing import (
        CubicBezierPathLike,
        Point2DLike,
        Point3D,
        Point3D_Array,
        Vector3D,
    )

__all__ = [
    "end_corner",
    "end_corner_index",
    "points_equal_2d",
    "start_corner",
    "subpath_split_indices",
    "unit_normal",
]


def points_equal_2d(p0: Point2DLike, p1: Point2DLike, tolerance: float) -> bool:
    """Whether two points are equal in ``x`` and ``y`` up to ``tolerance``.

    See :meth:`.VMobject.consider_points_equals_2d`.
    """
    rtol = 1.0e-5  # default from np.isclose()
    atol = tolerance

    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    if x0 != x1:
        x_diff = abs(x0 - x1)
        if not (x_diff <= atol + rtol * abs(x1) and math.isfinite(x_diff)):
            return False
    if y0 == y1:
        return True
    y_diff = abs(y0 - y1)
    return bool(y_diff <= atol + rtol * abs(y1) and math.isfinite(y_diff))


def subpath_split_indices(
    points: CubicBezierPathLike, tolerance: float, n_dims: int = 3, nppcc: int = 4
) -> npt.NDArray[np.int_]:
    """The ``[start, end)`` point ranges of the subpaths of ``points``.

    See :meth:`.VMobject.get_subpath_split_indices_from_points`.
    """
    points = np.asarray(points)
    n_pts = len(points)
    if n_pts < nppcc:
        return np.empty((0, 2), dtype=int)

    # Point indices where each new cubic curve starts.
    boundary_indices = np.arange(nppcc, n_pts, nppcc)
    if len(boundary_indices) == 0:
        # A single cubic curve: no internal boundaries to split on.
        return np.array([[0, n_pts]])

    # A boundary is a split where the previous curve's end anchor is not
    # close to the next curve's start anchor. This is an allocation-light
    # vectorization of np.isclose semantics: NaN is never close, while
    # infinities compare equal only when they have the same sign.
    rtol = 1.0e-5  # default from np.isclose()
    atol = tolerance
    ends = points[boundary_indices - 1, :n_dims]  # end of previous curve
    starts = points[boundary_indices, :n_dims]  # start of next curve
    with np.errstate(invalid="ignore"):
        diffs = np.abs(ends - starts)
        thresholds = atol + rtol * np.abs(starts)
        is_close = diffs <= thresholds
        finite_diffs = np.isfinite(diffs)
        if not finite_diffs.all():
            is_close &= finite_diffs
            is_close |= ends == starts
    is_split = ~np.all(is_close, axis=1)

    split_points = np.concatenate([[0], boundary_indices[is_split], [n_pts]])
    return np.stack([split_points[:-1], split_points[1:]], axis=1)


# The corners and normals that the 3D shading of a path uses (``shade_in_3d``).


def end_corner_index(points: Point3D_Array) -> int:
    """The index of the corner opposite the path's first point."""
    return ((len(points) - 1) // 6) * 3


def start_corner(points: Point3D_Array) -> Point3D:
    if len(points) == 0:
        return np.array(ORIGIN)
    return points[0]


def end_corner(points: Point3D_Array) -> Point3D:
    if len(points) == 0:
        return np.array(ORIGIN)
    return points[end_corner_index(points)]


def unit_normal(points: Point3D_Array, point_index: int) -> Vector3D:
    """The unit normal of the path at the point ``point_index``; ``UP`` where
    there is no plane to take one of.
    """
    n_points = len(points)
    # At most one curve: no plane to take a normal of.
    if n_points <= 4:
        return np.array(UP)
    i = point_index
    im3 = i - 3 if i > 2 else (n_points - 4)
    ip3 = i + 3 if i < (n_points - 3) else 3
    normal = get_unit_normal(points[ip3] - points[i], points[im3] - points[i])
    if np.linalg.norm(normal) == 0:
        return np.array(UP)
    return normal
