"""What both renderers need to know about the faces of a mesh snapshot.

A mesh is made of polygons (:attr:`.RMesh.faces`, quads for parametric surfaces, else
its triangles). The Cairo renderer draws each face as a path; the OpenGL renderer draws
the triangles, and outlines the faces in its shader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim.utils.bezier import interpolate

if TYPE_CHECKING:
    from typing import TypeAlias

    import numpy.typing as npt

    from manim.renderer.snapshot import RMesh
    from manim.typing import Face_Array, FloatRGBA_Array, Vector3D_Array

    #: Corners of faces, shape ``(m, k, 3)``: ``k`` points per face.
    FaceCorners: TypeAlias = npt.NDArray[np.float64]
    #: Outlines of faces, shape ``(m, 4 k, 3)``: one straight cubic curve per side.
    FaceOutlines: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "MESH_STROKE_WIDTH_UNIT",
    "face_colors",
    "face_normals",
    "face_outlines",
    "faces_of",
]

#: Scene units per unit of a mesh's ``stroke_width``, as for vectorized mobjects.
MESH_STROKE_WIDTH_UNIT = 0.01

#: Where the anchors and handles of a straight cubic curve lie.
_THIRDS = np.array([0, 1 / 3, 2 / 3, 1])


def faces_of(mesh: RMesh) -> Face_Array:
    """The polygons of ``mesh``, as vertex indices: its faces, or its triangles."""
    faces = np.asarray(mesh.faces)
    return faces if len(faces) else np.asarray(mesh.triangles)


def face_outlines(corners: FaceCorners) -> FaceOutlines:
    """The outlines of faces with corners of shape ``(m, k, 3)``: each side a
    straight cubic curve, as :meth:`.VMobject.set_points_as_corners` makes them.
    """
    following = np.roll(corners, -1, axis=1)
    sides: FaceOutlines = interpolate(
        corners[:, :, None, :], following[:, :, None, :], _THIRDS[:, None]
    )
    return sides.reshape(len(corners), -1, 3)


def face_normals(corners: FaceCorners) -> Vector3D_Array:
    """Normals of faces with corners of shape ``(m, k, 3)``, not normalized; zero for
    faces without area. For quads, the cross product of their diagonals.
    """
    if corners.shape[1] >= 4:
        return np.cross(corners[:, 2] - corners[:, 0], corners[:, 3] - corners[:, 1])
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def face_colors(mesh: RMesh, faces: Face_Array) -> FloatRGBA_Array:
    """One RGBA color per face: the texture at its middle, or the mean of its
    corners' colors.
    """
    rgbas = np.asarray(mesh.rgbas, dtype=float)
    count = len(mesh.points)
    texture = np.asarray(mesh.texture)
    if len(mesh.uvs) == count and texture.size:
        height, width = texture.shape[:2]
        uv = np.asarray(mesh.uvs)[faces].mean(axis=1)
        x = np.clip((uv[:, 0] * width).astype(int), 0, width - 1)
        y = np.clip((uv[:, 1] * height).astype(int), 0, height - 1)
        colors = texture[y, x].astype(float) / 255
        colors[:, 3] *= rgbas[:, 3].mean() if len(rgbas) != count else 1
        return colors
    if len(rgbas) == count:
        return rgbas[faces].mean(axis=1)
    return np.repeat(rgbas[:1], len(faces), axis=0)
