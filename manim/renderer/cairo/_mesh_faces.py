"""Meshes drawn by the Cairo renderer, as one filled polygon per face.

The faces of a mesh become path snapshots, sorted far to near; under a
:class:`.ThreeDCamera` they are then sorted with the other 3D paths of the scene. Each
face is colored once, lit at its middle by :func:`manim.renderer._lighting.shade`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from manim.appearance.material import UnlitMaterial
from manim.constants import CapStyleType, LineJointType
from manim.renderer._lighting import Light, is_lit, shade
from manim.renderer._mesh import face_colors, face_normals, face_outlines, faces_of
from manim.renderer._view import eye
from manim.renderer.snapshot import (
    STROKE_WIDTH_UNIT,
    RMesh,
    RNode,
    RPath,
    RSolidColor,
    RStroke,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manim.renderer.protocol import CameraState, LightState

__all__ = ["mesh_faces"]

_UNLIT = UnlitMaterial()
#: Points of a face's outline closer than this are taken as the same point, as
#: for vectorized mobjects (``tolerance_for_point_equality``).
_FACE_TOLERANCE = 1e-6


def mesh_faces(
    node: RNode,
    camera: CameraState,
    reference_pixel_size: tuple[int, int],
    lights: Sequence[LightState] = (),
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[RNode]:
    """Path snapshots for the faces of a mesh node, from far to near.

    Seams between faces are closed with outlines a pixel wide at
    ``reference_pixel_size`` (see :attr:`.RenderConfig.reference_pixel_size`).

    Each face node is keyed by the mesh node's key and the face's index, or has no key
    if the mesh has none.
    """
    mesh: RMesh = node.payload  # type: ignore[assignment]
    faces = faces_of(mesh)
    if len(faces) == 0 or len(mesh.points) == 0:
        return []
    corners = np.asarray(mesh.points)[faces]
    # The corners' mean: the middle of a face for its lighting and depth. (The
    # middle of its bounding box can be the same point for several faces: for a
    # tetrahedron around the origin, it is the origin for all four.)
    centers = corners.mean(axis=1)
    colors = face_colors(mesh, faces)
    if is_lit(mesh.material, "Cairo") and lights and not node.fixed_in_frame:
        colors[:, :3] = shade(
            colors[:, :3],
            face_normals(corners),
            centers,
            eye(camera),
            [Light(light.position, light.rgb, light.intensity) for light in lights],
            ambient,
            mesh.material,
        )
    outlines = face_outlines(corners)
    rotation = np.identity(3) if node.is_fixed_in_frame else camera.rotation
    depth = centers @ rotation[2]
    strokes = np.asarray(mesh.stroke_rgbas, dtype=float)
    if len(strokes) != len(faces):
        strokes = np.repeat(strokes[:1], len(faces), axis=0)
    # Without outlines, neighboring faces leave a seam where their antialiased edges
    # meet; opaque faces are outlined in their own color, a pixel wide, to close it.
    seam = (
        camera.default_frame_width / reference_pixel_size[0] / STROKE_WIDTH_UNIT
        if mesh.stroke_width == 0
        else None
    )
    versions = node.versions
    geometry = versions.get("geometry", 0)
    face_versions = {
        "geometry": geometry,
        "fill": max(
            versions.get(slot, 0)
            for slot in ("geometry", "rgbas", "uvs", "texture", "material")
        ),
        "stroke": max(geometry, versions.get("stroke", 0)),
        "material": 0,
        "depth_reference": geometry,
    }
    nodes: list[RNode] = []
    for index in np.argsort(depth, kind="stable"):
        fill = colors[index]
        stroke, width = strokes[index], mesh.stroke_width
        if seam is not None and fill[3] >= 1:
            stroke, width = fill, seam
        path = RPath(
            points=outlines[index],
            fill=RSolidColor(fill),
            stroke=RStroke(
                RSolidColor(stroke), width, LineJointType.AUTO, CapStyleType.AUTO
            ),
            background_stroke=None,
            material=_UNLIT,
            # Fixed faces keep the order above, after the scene's 3D paths.
            depth_reference=None if node.is_fixed_in_frame else centers[index],
            tolerance=1e-6,
        )
        nodes.append(
            RNode(
                # Distinct from mobject uids, which are not negative.
                uid=-((node.uid << 32) | int(index)) - 1,
                key=None if node.key is None else (node.key, int(index)),
                versions=face_versions,
                payload=path,
                placement=node.placement,
            )
        )
    return nodes
