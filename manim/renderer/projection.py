"""The projection of a camera with perspective, as a function of its state.

Cameras (:meth:`.ThreeDCamera.project_points`) and renderers share it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manim.renderer.protocol import CameraState
    from manim.typing import Point3D_Array

__all__ = ["project_points"]


def project_points(state: CameraState, points: Point3D_Array) -> Point3D_Array:
    """Rotate points about the frame center and scale x and y by the perspective
    factor ``focal_distance / (focal_distance - z)`` and the zoom.
    """
    focal_distance = state.focal_distance
    assert focal_distance is not None
    points = points - state.center
    points = np.dot(points, state.rotation.T)
    zs = points[:, 2]
    factor = focal_distance / (focal_distance - zs)
    factor[(focal_distance - zs) < 0] = 10**6
    scale = factor * state.zoom
    for i in 0, 1:
        points[:, i] *= scale
    return points
