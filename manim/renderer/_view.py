"""How a camera state places, orders and shades render snapshots: shared by the
renderers, which draw from a :class:`.CameraState` and never read cameras.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim.renderer.projection import project_points
from manim.renderer.snapshot import LegacyShaded, RMesh
from manim.utils._path_points import (
    end_corner,
    end_corner_index,
    start_corner,
    unit_normal,
)
from manim.utils.color import get_shaded_rgb

if TYPE_CHECKING:
    from manim.renderer.protocol import CameraState
    from manim.renderer.snapshot import RNode, RPath
    from manim.typing import FloatRGBA_Array, Point3D, Point3D_Array

__all__ = [
    "display_points",
    "eye",
    "is_3d",
    "place",
    "project_points",
    "shaded_rgbas",
    "sort_for_display",
    "view_transform_center",
]

#: Where a flat view is seen from: far above the frame, so that it looks straight
#: down on every point.
_FLAT_EYE_DISTANCE = 1e6


def is_3d(state: CameraState) -> bool:
    """Whether the state projects with perspective (a :class:`.ThreeDCamera`)."""
    return state.focal_distance is not None


def eye(state: CameraState) -> Point3D:
    """Where the scene is seen from, in scene coordinates."""
    distance = state.focal_distance
    if distance is None:
        distance = _FLAT_EYE_DISTANCE
    return np.asarray(state.center, dtype=float) + state.rotation.T @ [
        0.0,
        0.0,
        distance,
    ]


def view_transform_center(state: CameraState) -> Point3D:
    """The scene point at the middle of the image after :func:`place`.

    A perspective projection already subtracts the frame center.
    """
    return np.zeros(3) if is_3d(state) else np.asarray(state.center, dtype=float)


def place(
    state: CameraState,
    points: Point3D_Array,
    fixed_in_frame: bool,
    orientation_center: Point3D | None,
) -> Point3D_Array:
    """Where points are drawn, in the coordinates of :func:`view_transform_center`."""
    if not np.all(np.isfinite(points)):
        return np.zeros((1, 3))
    if not is_3d(state):
        if fixed_in_frame:
            # Drawn as if the frame were at its default position and size.
            scale = state.frame_width / state.default_frame_width
            return state.center + scale * points
        return points
    # The view transform is centered at the origin, so points fixed in frame are
    # drawn where they are.
    if fixed_in_frame:
        return points
    if orientation_center is not None:
        center = np.asarray(orientation_center).reshape((1, 3))
        return points + (project_points(state, center)[0] - orientation_center)
    return project_points(state, points)


def display_points(
    state: CameraState, node: RNode, points: Point3D_Array
) -> Point3D_Array:
    """:func:`place` for the points of a render snapshot."""
    return place(state, points, node.fixed_in_frame, node.fixed_orientation_center)


def sort_for_display(state: CameraState, nodes: list[RNode]) -> list[RNode]:
    """Order snapshots far to near for a perspective view, stably by their depth
    reference; nodes without one keep their order and come last, and so do nodes
    fixed in the frame, which have no place in the scene's depth. Flat views keep
    the order.
    """
    if not is_3d(state):
        return nodes
    depth_axis = state.rotation[2]

    def z_key(node: RNode) -> float:
        if node.fixed_in_frame:
            return np.inf  # type: ignore[no-any-return]
        payload = node.payload
        reference_point = getattr(payload, "depth_reference", None)
        if isinstance(payload, RMesh) and len(payload.points):
            # A whole mesh (on OpenGL; Cairo sorts its faces) goes by its middle.
            points = np.asarray(payload.points)
            reference_point = (points.min(axis=0) + points.max(axis=0)) / 2
        if reference_point is None:
            return np.inf  # type: ignore[no-any-return]
        return float(np.dot(reference_point, depth_axis))

    return sorted(nodes, key=z_key)


def shaded_rgbas(
    state: CameraState, path: RPath, rgbas: FloatRGBA_Array
) -> FloatRGBA_Array:
    """The stop colors of a path's paint after the shading of its material.

    :class:`.LegacyShaded` paths are shaded towards the light at their first and
    last corner, if the state shades and has a light; other paths keep ``rgbas``.
    """
    points = path.points
    light = state.light
    if not (
        state.shade_3d
        and light is not None
        and isinstance(path.material, LegacyShaded)
        and len(points) > 0
    ):
        return rgbas
    shaded = rgbas.repeat(2, axis=0) if len(rgbas) < 2 else np.array(rgbas[:2])
    shaded[0, :3] = get_shaded_rgb(
        shaded[0, :3], start_corner(points), unit_normal(points, 0), light
    )
    end = end_corner_index(points)
    shaded[1, :3] = get_shaded_rgb(
        shaded[1, :3], end_corner(points), unit_normal(points, end), light
    )
    return shaded
