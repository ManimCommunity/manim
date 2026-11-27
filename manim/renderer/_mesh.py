"""What the renderers need to know about the faces and vertices of a mesh snapshot.

A mesh is made of polygons (:attr:`.RMesh.faces`, quads for parametric surfaces, else
its triangles). The Cairo renderer draws each face as a path; the OpenGL renderer draws
the triangles, and outlines the faces in its shader; both, and the wgpu converter,
light smooth meshes by the normals at their vertices (:class:`NormalAccumulator`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse

from manim.utils.bezier import interpolate

if TYPE_CHECKING:
    from typing import TypeAlias

    import numpy.typing as npt

    from manim.renderer.snapshot import RMesh
    from manim.typing import (
        Face_Array,
        FloatRGBA_Array,
        Point3D_Array,
        TriangleIndices_Array,
        Vector3D_Array,
    )

    #: Corners of faces, shape ``(m, k, 3)``: ``k`` points per face.
    FaceCorners: TypeAlias = npt.NDArray[np.float64]
    #: Outlines of faces, shape ``(m, 4 k, 3)``: one straight cubic curve per side.
    FaceOutlines: TypeAlias = npt.NDArray[np.float64]

__all__ = [
    "NormalAccumulator",
    "face_colors",
    "face_normals",
    "face_outlines",
    "faces_of",
    "vertex_normals",
]

#: Where the anchors and handles of a straight cubic curve lie.
_THIRDS = np.array([0, 1 / 3, 2 / 3, 1])
#: Distance, relative to the size of a mesh, below which vertices are one point.
_WELD_TOLERANCE = 1e-6


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


class NormalAccumulator:
    """Computes the vertex normals of meshes with the same triangles quickly.

    It is made from a mesh's points and triangles and keeps which triangles each
    vertex is a corner of, with vertices at the same point welded (see
    :func:`vertex_normals`), as one sparse matrix: the normals of new points of the
    same triangles (a mesh that moves, rotates or scales) are then the triangles'
    normals times that matrix.

    Which vertices are welded is decided once, from the points it is made from.
    :meth:`fits` says whether that still holds for other points: not when welded
    vertices have moved apart (a deformed mesh). Vertices that come together
    later are not welded until an accumulator is made anew.
    """

    __slots__ = ("_corners", "_count", "_matrix", "_triangles", "_welded")

    def __init__(self, points: Point3D_Array, triangles: TriangleIndices_Array) -> None:
        points = np.asarray(points, dtype=float)
        triangles = np.asarray(triangles)
        count, triangle_count = len(points), len(triangles)
        self._triangles = triangles
        self._count = count
        self._corners = np.ascontiguousarray(triangles.T)
        rows = self._corners.ravel()
        welded = _welded(points, triangles)
        if welded is not None:
            rows = welded[rows]
        matrix = scipy.sparse.csr_matrix(
            (
                np.ones(3 * triangle_count),
                (rows, np.tile(np.arange(triangle_count), 3)),
            ),
            shape=(count, triangle_count),
        )
        #: The vertices welded to another one, and the ones they are welded to.
        self._welded: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]] | None = None
        if welded is not None:
            # Every vertex of a welded group gets the group's sum.
            matrix = scipy.sparse.csr_matrix(matrix[welded])
            moved = np.flatnonzero(welded != np.arange(count))
            self._welded = (moved, welded[moved])
        self._matrix = matrix

    def fits(self, points: Point3D_Array, triangles: TriangleIndices_Array) -> bool:
        """Whether it was made for these triangles and as many points, and the
        vertices it welded are still at the same points.
        """
        if triangles is not self._triangles or len(points) != self._count:
            return False
        if self._welded is None:
            return True
        points = np.asarray(points, dtype=float)
        vertices, representatives = self._welded
        size = float(np.ptp(points, axis=0).max()) or 1.0
        apart = np.abs(points[vertices] - points[representatives]).max()
        return bool(apart <= size * _WELD_TOLERANCE)

    def normals(self, points: np.ndarray) -> np.ndarray:
        """Unit normals at the vertices of ``points`` (with the triangles it was
        made for), as :func:`vertex_normals` gives them.
        """
        points = np.asarray(points, dtype=float)
        a, b, c = (points[corner] for corner in self._corners)
        u, v = b - a, c - a
        faces = np.empty_like(u)
        faces[:, 0] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        faces[:, 1] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        faces[:, 2] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        return _unit_or_up(np.asarray(self._matrix @ faces))


def vertex_normals(
    points: Point3D_Array, triangles: TriangleIndices_Array
) -> Vector3D_Array:
    """Unit normals at the vertices: the sum of the normals of their triangles,
    weighted by area. Vertices without a direction get ``(0, 0, 1)``.

    Vertices at the same point (up to :data:`_WELD_TOLERANCE` relative to the
    mesh's size) share their normal, so a surface whose parameter grid closes on
    itself (the seam of a torus or a sphere, the poles of a sphere) is lit
    without a visible seam. Meshes whose faces have corners of their own keep
    flat normals: their corners at one point belong to faces of their own.
    """
    corners = points[triangles]
    faces = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    normals = np.zeros_like(points, dtype=float)
    for i in range(3):
        np.add.at(normals, triangles[:, i], faces)
    welded = _welded(points, triangles)
    if welded is not None:
        sums = np.zeros_like(normals)
        np.add.at(sums, welded, normals)
        normals = sums[welded]
    return _unit_or_up(normals)


def _unit_or_up(normals: Vector3D_Array) -> Vector3D_Array:
    """``normals`` scaled to length one; ``(0, 0, 1)`` where they have no direction."""
    lengths = np.linalg.norm(normals, axis=1)
    flat = lengths < 1e-12
    normals[flat] = (0, 0, 1)
    lengths[flat] = 1
    return normals / lengths[:, None]


def _welded(
    points: Point3D_Array, triangles: TriangleIndices_Array
) -> npt.NDArray[np.intp] | None:
    """For each vertex, the index of one vertex at the same point, for vertices
    that share a point and are corners of the same smooth surface; ``None`` if no
    vertices share a point.

    A mesh whose triangles have corners of their own (every vertex in one
    triangle, or every quad's four corners its own) is left alone: it is meant
    to be lit flat.
    """
    if len(points) == 0:
        return None
    size = float(np.ptp(points, axis=0).max()) or 1.0
    keys = np.round(points / (size * _WELD_TOLERANCE)).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    if len(first) == len(points):
        return None
    # Flat meshes: no vertex is shared by triangles that are not one face's.
    uses = np.bincount(triangles.reshape(-1), minlength=len(points))
    if uses.max() <= 2:
        return None
    return first[inverse]
