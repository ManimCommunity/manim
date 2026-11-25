"""Draw render snapshots with OpenGL.

The OpenGL shaders draw quadratic Bézier curves; mobjects store cubic ones. Each path
is converted here, on the CPU, into quadratic pieces whose distance from the cubic
stays below a fraction of a pixel. The result is cached per mobject by the version of
its points, so a path that does not change is converted once.

Nodes are drawn in extraction order. Consecutive paths that cannot affect each other's
pixels are drawn together: one draw call for their background strokes, one winding
fill, one draw call for their strokes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import moderngl
import numpy as np
from PIL.Image import Resampling

from manim.appearance.material import PbrMaterial
from manim.renderer import _view
from manim.renderer._mesh import NormalAccumulator, faces_of
from manim.renderer._paint import paint_colors, shows
from manim.renderer._reuse import PrefixMemo, node_key, view_key
from manim.renderer.projection import project_points
from manim.renderer.protocol import CapabilityError
from manim.renderer.snapshot import (
    STROKE_WIDTH_UNIT,
    FacingCamera,
    RCameraView,
    RColorField,
    RImage,
    RMesh,
    RNode,
    RPath,
    RPoints,
    RSolidColor,
)

from ._winding_fill import WindingFill
from .shader import Shader

if TYPE_CHECKING:
    from manim.material import Material
    from manim.renderer.protocol import CameraState, LightState

    from .renderer import OpenGLRenderer, _CameraView

__all__ = ["DrawTarget", "SnapshotDrawing", "cubics_to_quadratics"]

#: Focal distance of a view without perspective.
_FLAT_FOCAL_DISTANCE = 1e6

#: Largest distance, in pixels, between a cubic curve and its quadratic pieces.
TOLERANCE_PIXELS = 0.1
#: Pieces per cubic at most; enough for any curve that fits on screen.
MAX_PIECES = 16

_STROKE_DTYPE = [
    ("point", np.float32, (3,)),
    ("prev_point", np.float32, (3,)),
    ("next_point", np.float32, (3,)),
    ("unit_normal", np.float32, (3,)),
    ("stroke_width", np.float32, (1,)),
    ("color", np.float32, (4,)),
]
_STROKE_FORMAT = "3f 3f 3f 3f 1f 4f"
#: The most lights the mesh shader takes (``MAX_LIGHTS`` in ``mesh/frag.glsl``).
MAX_LIGHTS = 8
#: The texture unit colour fields are bound to (images and meshes use unit 0).
_FIELD_TEXTURE_UNIT = 3
_POINT_DTYPE = [("point", np.float32, (3,)), ("color", np.float32, (4,))]
_MESH_DTYPE = np.dtype(
    [
        ("point", "f4", 3),
        ("normal", "f4", 3),
        ("color", "f4", 4),
        ("uv", "f4", 2),
        ("edge", "f4", 4),
        ("stroke", "f4", 4),
    ]
)
_MESH_FORMAT = "3f 3f 4f 2f 4f 4f"
_IMAGE_DTYPE = [("point", np.float32, (3,)), ("uv", np.float32, (2,))]
# Two triangles over an image's corners, stored as upper left, upper right, lower
# left, lower right; texture rows start at the top.
_QUAD_CORNERS = [0, 1, 2, 1, 3, 2]
_QUAD_UV = np.array([[0, 0], [1, 0], [0, 1], [1, 0], [1, 1], [0, 1]], dtype=np.float32)


def cubics_to_quadratics(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Approximate cubic Bézier curves by quadratic ones.

    Parameters
    ----------
    points
        Control points of ``n`` cubic curves, shape ``(4 * n, 3)``.
    tolerance
        Largest allowed distance between a cubic and its quadratic pieces.

    Returns
    -------
    np.ndarray
        Control points of the quadratic pieces, shape ``(m, 3, 3)``, in order. Each
        cubic is split into equal parameter intervals; their number follows from the
        error bound ``sqrt(3) / 36 * |P3 - 3 P2 + 3 P1 - P0| / k**3`` for ``k`` pieces.
    """
    return _convert(points, tolerance)[0]


def _convert(points: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic pieces of cubic curves, and the number of pieces per cubic."""
    cubics = points[: len(points) // 4 * 4].reshape(-1, 4, 3)
    if len(cubics) == 0:
        return np.empty((0, 3, 3)), np.zeros(0, dtype=int)
    p0, p1, p2, p3 = (cubics[:, i] for i in range(4))
    third_difference = np.linalg.norm(p3 - 3 * p2 + 3 * p1 - p0, axis=1)
    error = math.sqrt(3) / 36 * third_difference
    pieces = np.clip(np.ceil(np.cbrt(error / tolerance)), 1, MAX_PIECES).astype(int)

    curve = np.repeat(np.arange(len(cubics)), pieces)
    first_piece = np.repeat(np.cumsum(pieces) - pieces, pieces)
    index = np.arange(len(curve)) - first_piece
    count = pieces[curve]
    t0 = (index / count)[:, None]
    t1 = ((index + 1) / count)[:, None]
    controls = cubics[curve]
    q0 = _bezier(controls, t0)
    q3 = _bezier(controls, t1)
    span = (t1 - t0) / 3
    q1 = q0 + span * _bezier_derivative(controls, t0)
    q2 = q3 - span * _bezier_derivative(controls, t1)
    handle = (3 * (q1 + q2) - (q0 + q3)) / 4
    return np.stack((q0, handle, q3), axis=1), pieces


def _bezier(cubics: np.ndarray, t: np.ndarray) -> np.ndarray:
    s = 1 - t
    return (
        s**3 * cubics[:, 0]
        + 3 * s**2 * t * cubics[:, 1]
        + 3 * s * t**2 * cubics[:, 2]
        + t**3 * cubics[:, 3]
    )


def _bezier_derivative(cubics: np.ndarray, t: np.ndarray) -> np.ndarray:
    s = 1 - t
    return 3 * (
        s**2 * (cubics[:, 1] - cubics[:, 0])
        + 2 * s * t * (cubics[:, 2] - cubics[:, 1])
        + t**2 * (cubics[:, 3] - cubics[:, 2])
    )


class _Geometry(NamedTuple):
    """A path's quadratic pieces, with what drawing them needs."""

    version: int
    tolerance: float
    #: Quadratic control points, shape (m, 3, 3).
    quads: np.ndarray
    #: For each piece, the index of the first piece of its subpath.
    subpath_start: np.ndarray
    #: Bounding box: (low x, low y, low z, high x, high y, high z).
    box: np.ndarray


class _GeometryCache:
    """Quadratic geometry by uid, kept while the path is drawn in consecutive frames."""

    def __init__(self) -> None:
        self._current: dict[int, _Geometry] = {}
        self._previous: dict[int, _Geometry] = {}

    def next_frame(self) -> None:
        self._previous, self._current = self._current, {}

    def get(self, nodes: Sequence[RNode], tolerance: float) -> list[_Geometry]:
        """Geometry for each node; misses are converted in one batch.

        An entry is reused while the node's ``"geometry"`` slot (its points and
        point tolerance) keeps its version and the conversion tolerance is the same.
        """
        result: list[_Geometry | None] = []
        missing: list[int] = []
        for position, node in enumerate(nodes):
            version = node.versions.get("geometry", 0)
            entry = self._current.get(node.uid) or self._previous.get(node.uid)
            if (
                entry is not None
                and entry.version == version
                and entry.tolerance == tolerance
            ):
                self._current[node.uid] = entry
                result.append(entry)
            else:
                result.append(None)
                missing.append(position)
        if missing:
            self._convert([nodes[i] for i in missing], tolerance, missing, result)
        return result  # type: ignore[return-value]

    def _convert(
        self,
        nodes: list[RNode],
        tolerance: float,
        positions: list[int],
        result: list[_Geometry | None],
    ) -> None:
        paths: list[np.ndarray] = []
        for node in nodes:
            points = node.payload.points
            paths.append(points[: len(points) // 4 * 4])
        quads, pieces = _convert(
            np.concatenate(paths) if paths else np.empty((0, 3)), tolerance
        )
        # Split the batch back into paths: pieces per path from pieces per cubic.
        cubic_ends = np.cumsum([len(p) // 4 for p in paths])
        piece_ends = np.cumsum(pieces)[cubic_ends - 1] if len(pieces) else cubic_ends
        piece_ends = np.where(cubic_ends > 0, piece_ends, 0)
        piece_starts = np.r_[0, piece_ends[:-1]]
        starts = _batch_subpath_starts(quads, piece_starts, nodes)
        boxes = _batch_boxes(paths)
        for node, position, start, stop, box in zip(
            nodes, positions, piece_starts, piece_ends, boxes, strict=True
        ):
            entry = _Geometry(
                version=node.versions.get("geometry", 0),
                tolerance=tolerance,
                quads=quads[start:stop],
                subpath_start=starts[start:stop] - start,
                box=box,
            )
            self._current[node.uid] = entry
            result[position] = entry


def _batch_subpath_starts(
    quads: np.ndarray, piece_starts: np.ndarray, nodes: list[RNode]
) -> np.ndarray:
    """For each piece of a batch, the index of the first piece of its subpath."""
    if len(quads) == 0:
        return np.zeros(0, dtype=int)
    counts = np.diff(np.r_[piece_starts, len(quads)])
    tolerances = np.repeat([cast(RPath, n.payload).tolerance for n in nodes], counts)
    begins = np.ones(len(quads), dtype=bool)
    begins[1:] = (np.abs(quads[1:, 0] - quads[:-1, 2]) > tolerances[1:, None]).any(
        axis=1
    )
    begins[piece_starts[counts > 0]] = True
    indices = np.arange(len(quads))
    return np.maximum.accumulate(np.where(begins, indices, 0))


def _batch_boxes(paths: list[np.ndarray]) -> np.ndarray:
    """The bounding box of each path: (low x, low y, low z, high x, high y, high z)."""
    boxes = np.zeros((len(paths), 6))
    lengths = np.array([len(p) for p in paths])
    nonempty = np.flatnonzero(lengths)
    if len(nonempty):
        points = np.concatenate([paths[i] for i in nonempty])
        offsets = np.r_[0, np.cumsum(lengths[nonempty])[:-1]]
        boxes[nonempty, :3] = np.minimum.reduceat(points, offsets)
        boxes[nonempty, 3:] = np.maximum.reduceat(points, offsets)
    return boxes


def vertex_colors(
    rgbas: np.ndarray,
    gradient: np.ndarray | None,
    points: np.ndarray,
) -> np.ndarray:
    """Colors at ``points`` with Cairo's semantics.

    One color fills everything; several are stops of a linear gradient, evenly
    spaced from the gradient's start to its end point.
    """
    if len(rgbas) == 1 or gradient is None:
        return np.broadcast_to(rgbas[0], (len(points), 4))
    start, end = (np.asarray(p, dtype=float) for p in gradient)
    axis = end - start
    length_sq = float(np.dot(axis, axis))
    if length_sq == 0:
        return np.broadcast_to(rgbas[0], (len(points), 4))
    t = np.clip((points - start) @ axis / length_sq, 0, 1) * (len(rgbas) - 1)
    low = np.minimum(t.astype(int), len(rgbas) - 2)
    frac = (t - low)[:, None]
    return rgbas[low] * (1 - frac) + rgbas[low + 1] * frac


class DrawTarget(NamedTuple):
    """Where and from which camera to draw."""

    framebuffer: moderngl.Framebuffer
    #: Width and height in pixels.
    size: tuple[int, int]
    camera: CameraState
    view: _CameraView
    #: Camera uniforms of the shaders (see ``OpenGLRenderer.camera_uniforms``).
    uniforms: dict[str, Any]
    #: See :attr:`.RenderConfig.reference_pixel_size`.
    reference_size: tuple[int, int]
    #: See :attr:`.RenderConfig.lights` and :attr:`.RenderConfig.ambient`.
    lights: tuple[LightState, ...] = ()
    ambient: tuple[float, float, float] = (0.0, 0.0, 0.0)


class _Group(NamedTuple):
    nodes: list[RNode]
    geometry: list[_Geometry]
    #: Whether any member has a fill.
    filled: bool
    #: Pixel box containing what the group draws: (low x, low y, high x, high y).
    box: np.ndarray


class _Screen(NamedTuple):
    """Projects scene points to pixels, as the shaders do (y up, origin bottom left)."""

    center: np.ndarray
    rotation: np.ndarray
    focal_distance: float
    #: Pixels per scene unit in x and y.
    scale: np.ndarray
    size: np.ndarray

    @classmethod
    def of(cls, target: DrawTarget) -> _Screen:
        view = target.view
        size = np.array(target.size, dtype=float)
        return cls(
            center=view.center,
            rotation=view.rotation,
            focal_distance=view.focal_distance,
            scale=size / np.array(view.frame_shape),
            size=size,
        )

    def boxes(self, corners: np.ndarray, margin: np.ndarray) -> np.ndarray:
        """Pixel boxes of point sets, shape ``(n, k, 3)``, widened by ``margin`` pixels.

        A set with a point at or behind the camera covers the whole screen.
        """
        points = (corners - self.center) @ self.rotation.T
        depth = self.focal_distance - points[..., 2]
        behind = (depth <= 0).any(axis=-1)
        factor = self.focal_distance / np.where(depth > 0, depth, 1)
        pixels = points[..., :2] * factor[..., None] * self.scale + self.size / 2
        low = pixels.min(axis=-2) - margin[:, None]
        high = pixels.max(axis=-2) + margin[:, None]
        boxes = np.concatenate((low, high), axis=-1)
        boxes[behind] = (0, 0, *self.size)
        return boxes

    def scissor(self, box: np.ndarray) -> tuple[int, int, int, int]:
        """The pixel rectangle ``(x, y, width, height)`` covering ``box``."""
        x0, y0 = np.maximum(np.floor(box[:2]), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(box[2:]), self.size).astype(int)
        return int(x0), int(y0), max(int(x1 - x0), 0), max(int(y1 - y0), 0)


class SnapshotDrawing:
    """Draws render snapshots into the renderer's frame buffer."""

    def __init__(self, renderer: OpenGLRenderer) -> None:
        self._renderer = renderer
        self._geometry = _GeometryCache()
        self._fill = WindingFill(renderer)
        self._textures = _TextureCache(renderer)
        self._normals = _NormalCache()
        #: Textures of colour fields by grid id (premultiplied RGBA).
        self._fields: dict[int, moderngl.Texture] = {}
        self._layer: _Layer | None = None
        self._memo = PrefixMemo()

    def release(self) -> None:
        self._fill.release()
        self._textures.release()
        for texture in self._fields.values():
            if not isinstance(texture.mglo, moderngl.InvalidObject):
                texture.release()
        self._fields.clear()
        if self._layer is not None:
            self._layer.release()
            self._layer = None
        self._memo.clear()

    def draw(
        self,
        nodes: Iterable[RNode],
        target: DrawTarget,
        views: dict[Any, moderngl.Texture] | None = None,
        background: Any = None,
    ) -> None:
        """Draw ``nodes`` in order into ``target``, which holds only its background.

        Consecutive paths are batched. ``views`` holds the pixels of camera views
        (``RCameraView`` nodes) by uid; ``background`` identifies what the target
        was cleared to.

        The drawing of the leading nodes that did not change since the previous
        frame is kept in a layer and copied instead of drawn again while they and
        the view stay the same (see :mod:`manim.renderer._reuse`).
        """
        self._geometry.next_frame()
        self._textures.next_frame()
        self._normals.next_frame()
        nodes = list(nodes)
        keys = [node_key(node) for node in nodes]
        view = view_key(
            target.camera,
            target.size,
            target.reference_size,
            background,
            target.lights,
            target.ambient,
        )
        memo = self._memo
        context = self._renderer.context
        start = memo.count_reusable(keys, view)
        if start:
            assert self._layer is not None
            context.copy_framebuffer(target.framebuffer, self._layer.framebuffer)
        else:
            start = memo.count_worth_saving(keys, view)
            if start:
                layer = self._layer
                if layer is None or layer.size != target.size:
                    if layer is not None:
                        layer.release()
                    layer = self._layer = _Layer(self._renderer, target)
                context.copy_framebuffer(layer.framebuffer, target.framebuffer)
                self._draw_nodes(
                    nodes[:start], target._replace(framebuffer=layer.framebuffer), views
                )
                memo.mark_saved(keys[:start], view)
                context.copy_framebuffer(target.framebuffer, layer.framebuffer)
            else:
                memo.forget_saved()
        self._draw_nodes(nodes[start:], target, views)
        memo.record_drawn(keys, view)

    def _draw_nodes(
        self,
        nodes: list[RNode],
        target: DrawTarget,
        views: dict[Any, moderngl.Texture] | None,
    ) -> None:
        if not nodes:
            return
        self._base_target = target
        self._placed_targets: dict[str, DrawTarget] = {"normal": target}
        paths = [node for node in nodes if isinstance(node.payload, RPath)]
        self._target = target
        geometry = iter(self._geometry.get(paths, self._tolerance()) if paths else ())
        run: list[RNode] = []
        run_geometry: list[_Geometry] = []
        run_kind = "normal"
        for node in nodes:
            kind, shift = self._placement(node)
            if isinstance(node.payload, RPath):
                geo = next(geometry)
                if shift is not None:
                    geo = geo._replace(
                        quads=geo.quads + shift, box=geo.box + np.r_[shift, shift]
                    )
                if kind != run_kind:
                    self._draw_paths(run, run_geometry, run_kind)
                    run, run_geometry, run_kind = [], [], kind
                run.append(node)
                run_geometry.append(geo)
                continue
            self._draw_paths(run, run_geometry, run_kind)
            run, run_geometry = [], []
            self._target = self._placed_target(kind)
            payload = node.payload
            if shift is not None:
                payload = payload._replace(points=payload.points + shift)
            if isinstance(payload, RPoints):
                self._draw_points(payload)
            elif isinstance(payload, RMesh):
                self._draw_mesh(node, payload)
            elif isinstance(payload, RImage):
                self._draw_image(node, payload)
            elif isinstance(payload, RCameraView) and views:
                # Copies of a view show the picture of the same camera.
                texture = views.get(node.uid, views.get(("camera", payload.camera_id)))
                if texture is not None:
                    # Rendered textures start at the bottom row.
                    self._draw_quad(payload.points, texture, bottom_up=True)
        self._draw_paths(run, run_geometry, run_kind)

    def _placement(self, node: RNode) -> tuple[str, np.ndarray | None]:
        """How a node is placed on screen, and the shift of its points if any.

        Fixed-in-frame nodes are drawn as from a camera at its default frame. Under a
        :class:`.ThreeDCamera`, nodes with a fixed orientation move with their
        projected center but are not rotated, as on the Cairo renderer.
        """
        if node.is_fixed_in_frame:
            return "frame", None
        camera = self._base_target.camera
        placement = node.placement
        if isinstance(placement, FacingCamera) and _view.is_3d(camera):
            center = placement.center
            projected = project_points(camera, np.asarray(center).reshape((1, 3)))[0]
            return "frame", projected - center
        return "normal", None

    def _placed_target(self, kind: str) -> DrawTarget:
        target = self._placed_targets.get(kind)
        if target is None:
            base = self._base_target
            camera = base.camera
            if _view.is_3d(camera):
                # Its view transform is centered at the origin with its frame size.
                shape = (camera.frame_width, camera.frame_height)
            else:
                width = camera.default_frame_width
                shape = (width, width * base.size[1] / base.size[0])
            view = base.view._replace(
                center=np.zeros(3),
                rotation=np.identity(3),
                frame_shape=shape,
                focal_distance=_FLAT_FOCAL_DISTANCE,
            )
            target = base._replace(
                view=view, uniforms=self._renderer.camera_uniforms(view, base.size[1])
            )
            self._placed_targets[kind] = target
        return target

    def _draw_paths(
        self, nodes: list[RNode], geometry: list[_Geometry], kind: str
    ) -> None:
        if not nodes:
            return
        self._target = self._placed_target(kind)
        #: How the paths being drawn are placed (colour fields need "normal").
        self._kind = kind
        screen = _Screen.of(self._target)
        for group in _groups(nodes, geometry, screen):
            self._draw_group(group, screen)

    def _draw_points(self, cloud: RPoints) -> None:
        if len(cloud.points) == 0:
            return
        vertices = np.empty(len(cloud.points), dtype=_POINT_DTYPE)
        vertices["point"] = cloud.points
        rgbas = np.asarray(cloud.rgbas)
        vertices["color"] = (
            rgbas
            if len(rgbas) == len(cloud.points)
            else np.resize(rgbas, (len(cloud.points), 4))
        )
        self._renderer.context.enable(moderngl.PROGRAM_POINT_SIZE)
        self._pass(
            "point_cloud",
            vertices,
            "3f 4f",
            moderngl.POINTS,
            point_size=float(max(1, int(self._point_thickness(cloud.stroke_width)))),
        )

    def _point_thickness(self, stroke_width: float) -> float:
        """Point size in pixels: ``stroke_width`` is meant at the reference size,
        as on the Cairo renderer.
        """
        base = sum(self._target.reference_size)
        return 1 + (stroke_width - 1) * base / sum(self._target.size)

    def _draw_mesh(self, node: RNode, mesh: RMesh) -> None:
        count = len(mesh.points)
        if count == 0 or len(mesh.triangles) == 0:
            return
        if node.is_fixed_in_frame and mesh.depth_test:
            # A fixed mesh hides its own triangles only: it neither tests against
            # nor leaves the depth of the scene or of other fixed meshes. Fixed
            # nodes are drawn last, so nothing needs the depth cleared here.
            self._clear_depth()
        texture = None
        if len(mesh.uvs) == count:
            texture = self._textures.get(node, "texture", mesh.texture)
        if texture is not None:
            texture.use(0)
        if mesh.stroke_width > 0:
            vertices, indices = self._faceted(node, mesh, texture is not None)
        else:
            vertices = np.zeros(count, dtype=_MESH_DTYPE)
            vertices["point"] = mesh.points
            vertices["normal"] = self._normals.get(node, mesh)
            rgbas = np.asarray(mesh.rgbas)
            vertices["color"] = (
                rgbas if len(rgbas) == count else np.resize(rgbas, (count, 4))
            )
            if texture is not None:
                vertices["uv"] = mesh.uvs
            vertices["edge"] = 1
            indices = np.ascontiguousarray(mesh.triangles, dtype="i4")
        if mesh.depth_test and (vertices["color"][:, 3] < 1).any():
            # Translucent: show only the front-most layer of the mesh, so that how
            # its triangles are ordered does not show.
            framebuffer = self._target.framebuffer
            framebuffer.color_mask = False, False, False, False
            try:
                self._pass(
                    "mesh", vertices, _MESH_FORMAT, moderngl.TRIANGLES, indices, True
                )
            finally:
                framebuffer.color_mask = True, True, True, True
        # Outlines are as wide as a stroke on the Cairo renderer, half in each face.
        pixels_per_unit = self._target.size[1] / self._target.view.frame_shape[1]
        self._pass(
            "mesh",
            vertices,
            _MESH_FORMAT,
            moderngl.TRIANGLES,
            indices=indices,
            depth_test=mesh.depth_test,
            has_texture=float(texture is not None),
            image=0,
            # The shader measures from the edge to each side: half the width.
            edge_width=0.5
            * STROKE_WIDTH_UNIT
            * float(mesh.stroke_width)
            * pixels_per_unit,
            **self._lighting(node, mesh.material),
        )

    def _faceted(
        self, node: RNode, mesh: RMesh, textured: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vertices of each triangle of its own, so that the shader can outline the
        faces (polygons as fans of triangles), and the indices of the triangles.

        Each corner carries its barycentric coordinates in its triangle (``edge``,
        0 on the side opposite the corner); a side of the triangle that is not a
        side of the face (inside a fan) gets 1 at every corner, so it is never
        outlined.
        """
        faces = faces_of(mesh)
        count, k = faces.shape
        fan = np.array([[0, i, i + 1] for i in range(1, k - 1)])
        # Corner indices into the mesh, three per triangle, triangles of a face
        # in order.
        corners = faces[:, fan].reshape(-1)
        vertices = np.zeros(len(corners), dtype=_MESH_DTYPE)
        vertices["point"] = np.asarray(mesh.points)[corners]
        vertices["normal"] = self._normals.get(node, mesh)[corners]
        rgbas = np.asarray(mesh.rgbas)
        vertices["color"] = (
            rgbas[corners] if len(rgbas) == len(mesh.points) else rgbas[0]
        )
        strokes = np.asarray(mesh.stroke_rgbas)
        per_face = 3 * (k - 2)
        vertices["stroke"] = (
            np.repeat(strokes, per_face, axis=0)
            if len(strokes) == count
            else strokes[0]
        )
        if textured:
            vertices["uv"] = np.asarray(mesh.uvs)[corners]
        edges = np.ones((k - 2, 3, 4), dtype=np.float32)
        for t, i in enumerate(range(1, k - 1)):
            # The triangle (0, i, i + 1): the side opposite corner 0 is a side of
            # the face; (0, i + 1) only for the last, (0, i) only for the first.
            sides = (True, i + 1 == k - 1, i == 1)
            for corner in range(3):
                for opposite, is_side in enumerate(sides):
                    if is_side:
                        edges[t, corner, opposite] = float(corner == opposite)
        vertices["edge"] = np.tile(edges.reshape(-1, 4), (count, 1))
        indices = np.arange(len(corners), dtype="i4")
        return vertices, indices

    def _lighting(self, node: RNode, material: Material) -> dict[str, Any]:
        """The lighting uniforms of the mesh shader (see ``include/lighting.glsl``).

        Meshes are lit by the frame's lights; without lights, and when fixed in
        frame, they show their base colors.
        """
        target = self._base_target
        lights = target.lights
        if not isinstance(material, PbrMaterial) or not lights or node.fixed_in_frame:
            return {"lit": 0.0}
        if len(lights) > MAX_LIGHTS:
            raise CapabilityError(
                f"The OpenGL renderer draws at most {MAX_LIGHTS} lights; the frame "
                f"has {len(lights)}."
            )
        padding = MAX_LIGHTS - len(lights)
        return {
            "lit": 1.0,
            "roughness": material.roughness,
            "metallic": material.metallic,
            "emissive": material.emissive,
            "ambient": tuple(target.ambient),
            "light_count": len(lights),
            "light_positions": tuple(
                [tuple(map(float, light.position)) for light in lights]
                + [(0.0, 0.0, 0.0)] * padding
            ),
            "light_rgbs": tuple(
                [tuple(light.rgb) for light in lights] + [(0.0, 0.0, 0.0)] * padding
            ),
            "light_intensities": tuple(
                [float(light.intensity) for light in lights] + [0.0] * padding
            ),
            "eye_position": tuple(target.view.eye),
        }

    def _draw_image(self, node: RNode, image: RImage) -> None:
        texture = self._textures.get(
            node,
            "pixels",
            image.pixel_array,
            nearest=image.resampling_algorithm == Resampling.NEAREST,
        )
        if texture is None:
            return
        self._draw_quad(image.points, texture)

    def _draw_quad(
        self, points: np.ndarray, texture: moderngl.Texture, bottom_up: bool = False
    ) -> None:
        """Draw a texture onto four corners (upper left, upper right, lower left,
        lower right). Its rows start at the top unless ``bottom_up``.
        """
        vertices = np.empty(6, dtype=_IMAGE_DTYPE)
        vertices["point"] = np.asarray(points, dtype=float)[_QUAD_CORNERS]
        vertices["uv"] = _QUAD_UV
        if bottom_up:
            vertices["uv"][:, 1] = 1 - _QUAD_UV[:, 1]
        texture.use(0)
        self._pass("image_quad", vertices, "3f 2f", moderngl.TRIANGLES, image=0)

    def _clear_depth(self) -> None:
        framebuffer = self._target.framebuffer
        if framebuffer.depth_attachment is None:
            return
        framebuffer.color_mask = False, False, False, False
        try:
            framebuffer.clear(depth=1.0)
        finally:
            framebuffer.color_mask = True, True, True, True

    def _pass(
        self,
        shader_name: str,
        vertices: np.ndarray,
        layout: str,
        primitive: int,
        indices: np.ndarray | None = None,
        depth_test: bool = False,
        **uniforms: float,
    ) -> None:
        """Draw ``vertices`` with a shader, blending into the frame.

        ``indices`` picks vertices by index. With ``depth_test``, fragments behind
        earlier ones drawn with a depth test are hidden.
        """
        renderer = self._renderer
        context = renderer.context
        shader = Shader(context, name=shader_name)
        for name, value in self._target.uniforms.items():
            shader.set_uniform(name, value)
        for name, value in uniforms.items():
            shader.set_uniform(name, value)
        buffer = context.buffer(vertices.tobytes())
        index_buffer = None if indices is None else context.buffer(indices.tobytes())
        try:
            array = context.vertex_array(
                shader.shader_program,
                [(buffer, layout, *vertices.dtype.names)],
                index_buffer=index_buffer,
            )
            try:
                if depth_test:
                    context.enable(moderngl.DEPTH_TEST)
                    context.depth_func = "<="
                else:
                    context.disable(moderngl.DEPTH_TEST)
                context.enable(moderngl.BLEND)
                context.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                self._target.framebuffer.use()
                array.render(primitive)
            finally:
                array.release()
                context.disable(moderngl.DEPTH_TEST)
        finally:
            buffer.release()
            if index_buffer is not None:
                index_buffer.release()

    def _tolerance(self) -> float:
        """The conversion tolerance in scene units, rounded down to a power of two.

        Rounding keeps cached geometry valid while the camera zooms a little.
        """
        frame_width = self._target.view.frame_shape[0]
        tolerance = TOLERANCE_PIXELS * frame_width / self._target.size[0]
        return 2.0 ** math.floor(math.log2(tolerance))

    def _field_uniforms(self, paint: Any, kind: str) -> dict[str, Any]:
        """The uniforms that bind ``paint`` in the shaders: off unless it is a
        colour field, which is sampled at each fragment's scene position.

        ``kind`` is how the path is placed (see :meth:`_placement`).

        Raises
        ------
        CapabilityError
            For a field under a perspective camera, not in a plane parallel to the
            frame, or on a mobject fixed in the frame or in its orientation.
        """
        if not isinstance(paint, RColorField):
            return {"field_on": 0.0}
        target = self._target
        corners = np.asarray(paint.corners, dtype=float)
        if (
            _view.is_3d(target.camera)
            or kind != "normal"
            or not np.allclose(corners[:, 2], corners[0, 2])
        ):
            raise CapabilityError(
                "The OpenGL renderer draws colour fields only in a plane parallel to "
                "the frame, under a flat camera, on mobjects that are not fixed in "
                "the frame."
            )
        texture = self._fields.get(paint.grid_id)
        if texture is None:
            grid = np.asarray(paint.grid, dtype=float)
            rows, columns = grid.shape[:2]
            premultiplied = np.concatenate(
                [grid[..., :3] * grid[..., 3:4], grid[..., 3:4]], axis=-1
            )
            data = np.ascontiguousarray(np.round(premultiplied * 255), dtype=np.uint8)
            texture = self._renderer.context.texture((columns, rows), 4, data.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            texture.repeat_x = texture.repeat_y = False
            self._fields[paint.grid_id] = texture
            self._renderer._resources.callback(_release_texture, texture)
        texture.use(_FIELD_TEXTURE_UNIT)
        view = target.view
        width, height = target.size
        frame_width, frame_height = view.frame_shape
        # Fragment position (pixels from the bottom left) -> scene point -> texel.
        to_scene = np.array(
            [
                [frame_width / width, 0, view.center[0] - frame_width / 2],
                [0, frame_height / height, view.center[1] - frame_height / 2],
                [0, 0, 1],
            ]
        )
        upper_left, upper_right, lower_left = corners
        eu, ev = (upper_right - upper_left)[:2], (lower_left - upper_left)[:2]
        u, v = eu / (eu @ eu), ev / (ev @ ev)
        to_texel = np.array(
            [[*u, -u @ upper_left[:2]], [*v, -v @ upper_left[:2]], [0, 0, 1]]
        )
        matrix = to_texel @ to_scene
        return {
            "field_on": 1.0,
            "field_texture": _FIELD_TEXTURE_UNIT,
            # GLSL matrices are column-major.
            "field_uv": tuple(matrix.T.flatten()),
            "field_opacity": paint.opacity,
        }

    def _draw_group(self, group: _Group, screen: _Screen) -> None:
        if any(_background_width(n.payload) > 0 for n in group.nodes):  # type: ignore[arg-type]
            self._draw_strokes(group, background=True)
        if group.filled:
            first: RPath = group.nodes[0].payload  # type: ignore[assignment]
            self._fill.draw(
                group.nodes,
                group.geometry,
                self._target,
                screen.scissor(group.box),
                self._field_uniforms(first.fill, self._kind),
            )
        if any(n.payload.stroke.width > 0 for n in group.nodes):  # type: ignore[union-attr]
            self._draw_strokes(group, background=False)

    def _draw_strokes(self, group: _Group, background: bool) -> None:
        geometry, widths, colours = [], [], []
        for node, geo in zip(group.nodes, group.geometry, strict=True):
            path: RPath = node.payload  # type: ignore[assignment]
            stroke = path.background_stroke if background else path.stroke
            if stroke is None or stroke.width == 0 or len(geo.quads) == 0:
                continue
            geometry.append(geo)
            widths.append(stroke.width)
            colours.append(
                # Strokes are not lit.
                paint_colors(stroke.paint)
            )
        if not geometry:
            return
        vertices = _stroke_vertices(geometry, widths, colors)
        first: RPath = group.nodes[0].payload  # type: ignore[assignment]
        first_stroke = first.background_stroke if background else first.stroke
        self._pass(
            "quadratic_bezier_stroke",
            vertices,
            _STROKE_FORMAT,
            moderngl.TRIANGLES,
            joint_type=float(first.stroke.joint.value),
            flat_stroke=0.0,
            **self._field_uniforms(
                None if first_stroke is None else first_stroke.paint, self._kind
            ),
        )


def _stroke_vertices(
    geometry: list[_Geometry],
    widths: list[float],
    colors: list[tuple[np.ndarray, np.ndarray | None]],
) -> np.ndarray:
    """Stroke shader input for the paths of one group, built in one pass."""
    quads, counts, offsets = _concatenate(geometry)
    total = len(quads)
    path_of_piece = np.repeat(np.arange(len(counts)), counts)
    offset = offsets[path_of_piece]
    count = counts[path_of_piece]
    local = np.arange(total) - offset
    vertices = np.empty(3 * total, dtype=_STROKE_DTYPE)
    points = quads.reshape(-1, 3)
    vertices["point"] = points
    vertices["prev_point"] = quads[offset + (local - 1) % count].reshape(-1, 3)
    vertices["next_point"] = quads[offset + (local + 1) % count].reshape(-1, 3)
    vertices["unit_normal"] = (0.0, 0.0, 1.0)
    vertices["stroke_width"] = np.repeat(widths, 3 * counts)[:, None]
    vertices["color"] = _colors(colors, points, 3 * counts)
    return vertices


def _concatenate(
    geometry: Sequence[_Geometry],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pieces of several paths, pieces per path and each path's first piece."""
    counts = np.array([len(g.quads) for g in geometry])
    quads = np.concatenate([g.quads for g in geometry])
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    return quads, counts, offsets


def _colors(
    colors: list[tuple[np.ndarray, np.ndarray | None]],
    points: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Vertex colors of several paths; ``counts`` vertices per path."""
    first = np.array([rgbas[0] for rgbas, _ in colors])
    result = np.repeat(first, counts, axis=0)
    ends = np.cumsum(counts)
    for (rgbas, gradient), end, count in zip(colors, ends, counts, strict=True):
        if len(rgbas) > 1 and gradient is not None:
            start = end - count
            result[start:end] = vertex_colors(rgbas, gradient, points[start:end])
    return result


def _groups(
    nodes: list[RNode], geometry: list[_Geometry], screen: _Screen
) -> list[_Group]:
    """Split paths into runs that can be drawn together.

    A run is drawn in three passes: all background strokes, all fills, all strokes.
    That gives what drawing its paths one by one gives unless an earlier path's fill
    or stroke overlaps a later path's background stroke or fill on screen. So a path
    with only a stroke always joins; a path with a fill or background stroke joins
    if its pixel box is disjoint from what earlier members draw (their whole box if
    they are filled, otherwise their stroke pieces). Paths of one color without
    strokes, such as the glyphs of a text, may also overlap each other: their union
    is colored once. For opaque paths that gives the same pixels; for translucent
    ones it differs where their shapes overlap, which glyphs rarely do.
    """
    paths: list[RPath] = [n.payload for n in nodes]  # type: ignore[misc]
    widths = np.array([max(p.stroke.width, _background_width(p)) for p in paths])
    # Antialiasing reaches about a pixel and a half beyond an edge.
    margins = 0.005 * widths * screen.scale.max() + 2
    corners = np.array([g.box for g in geometry])[:, _CORNERS]
    boxes = screen.boxes(corners, margins)
    filled = np.array([shows(p.fill) or _background_width(p) > 0 for p in paths])
    piece_boxes, piece_offsets = _piece_boxes(geometry, ~filled, screen, margins)

    # Members, whether any is filled, and the pixel box of each group so far.
    groups: list[tuple[list[RNode], list[_Geometry], list[Any]]] = []
    regions = _Boxes()
    color: tuple[float, ...] | None = None
    joint = None
    binding: Any = None
    box_list = boxes.tolist()
    filled_list = filled.tolist()
    for index, (node, geo, path) in enumerate(zip(nodes, geometry, paths, strict=True)):
        x0, y0, x1, y1 = box_list[index]
        is_filled = filled_list[index]
        single = (
            tuple(path.fill.rgba)
            if isinstance(path.fill, RSolidColor)
            and not isinstance(path.material, PbrMaterial)
            and path.stroke.width == 0
            else None
        )
        # Paths of one group bind the same colour fields (or none).
        joins = (
            bool(groups) and path.stroke.joint == joint and _binding(path) == binding
        )
        if joins and is_filled:
            same_color = single is not None and single == color
            gx0, gy0, gx1, gy1, _ = groups[-1][2]
            joins = same_color or not (
                gx0 < x1
                and gx1 > x0
                and gy0 < y1
                and gy1 > y0
                and regions.overlaps(boxes[index])
            )
        if is_filled or len(geo.quads) == 0:
            region = boxes[index : index + 1]
        else:
            region = piece_boxes[piece_offsets[index] : piece_offsets[index + 1]]
        if joins:
            members, members_geometry, state = groups[-1]
            members.append(node)
            members_geometry.append(geo)
            if is_filled and not state[4]:
                color = single
            state[0] = min(state[0], x0)
            state[1] = min(state[1], y0)
            state[2] = max(state[2], x1)
            state[3] = max(state[3], y1)
            state[4] = state[4] or is_filled
        else:
            groups.append(([node], [geo], [x0, y0, x1, y1, is_filled]))
            regions = _Boxes()
            color = single
            joint = path.stroke.joint
            binding = _binding(path)
        regions.add(region)
    return [
        _Group(members, members_geometry, state[4], np.array(state[:4]))
        for members, members_geometry, state in groups
    ]


def _binding(path: RPath) -> Any:
    """The colour fields a path's paints bind; a group's paths share them."""

    def of(paint: Any) -> Any:
        if isinstance(paint, RColorField):
            return (paint.grid_id, np.asarray(paint.corners).tobytes(), paint.opacity)
        return None

    background = path.background_stroke
    return (
        of(path.fill),
        of(path.stroke.paint),
        of(background.paint) if background is not None else None,
    )


def _release_texture(texture: moderngl.Texture) -> None:
    if not isinstance(texture.mglo, moderngl.InvalidObject):
        texture.release()


def _piece_boxes(
    geometry: list[_Geometry], wanted: np.ndarray, screen: _Screen, margins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pixel boxes of the pieces of the wanted paths, projected in one batch.

    The boxes of path ``i`` are ``boxes[offsets[i] : offsets[i + 1]]``.
    """
    counts = np.array(
        [len(g.quads) if w else 0 for g, w in zip(geometry, wanted, strict=True)]
    )
    offsets = np.r_[0, np.cumsum(counts)]
    if offsets[-1] == 0:
        return np.empty((0, 4)), offsets
    quads = np.concatenate(
        [g.quads for g, c in zip(geometry, counts, strict=True) if c]
    )
    return screen.boxes(quads, np.repeat(margins, counts)), offsets


class _Boxes:
    """A growing array of pixel boxes, tested against one box at a time."""

    def __init__(self) -> None:
        self._boxes = np.empty((16, 4))
        self._count = 0

    def add(self, boxes: np.ndarray) -> None:
        end = self._count + len(boxes)
        if end > len(self._boxes):
            grown = np.empty((max(2 * len(self._boxes), end), 4))
            grown[: self._count] = self._boxes[: self._count]
            self._boxes = grown
        self._boxes[self._count : end] = boxes
        self._count = end

    def overlaps(self, box: np.ndarray) -> bool:
        current = self._boxes[: self._count]
        return bool(
            (
                (current[:, 0] < box[2])
                & (current[:, 2] > box[0])
                & (current[:, 1] < box[3])
                & (current[:, 3] > box[1])
            ).any()
        )


#: Indices of the eight corners in a box (low x, low y, low z, high x, high y, high z).
_CORNERS = np.array(np.meshgrid([0, 3], [1, 4], [2, 5], indexing="ij")).reshape(3, -1).T


def _background_width(path: RPath) -> float:
    return 0.0 if path.background_stroke is None else path.background_stroke.width


class _Layer:
    """An offscreen copy of a target with the drawing of some nodes."""

    def __init__(self, renderer: OpenGLRenderer, target: DrawTarget) -> None:
        context = renderer.context
        self.size = target.size
        self.texture = context.texture(self.size, components=4)
        self.depth = (
            context.depth_renderbuffer(self.size)
            if target.framebuffer.depth_attachment is not None
            else None
        )
        self.framebuffer = context.framebuffer([self.texture], self.depth)
        renderer._resources.callback(self.release)

    def release(self) -> None:
        """Release the GPU objects; safe to call more than once."""
        for resource in (self.framebuffer, self.texture, self.depth):
            if resource is not None and not isinstance(
                resource.mglo, moderngl.InvalidObject
            ):
                resource.release()


class _NormalCache:
    """Vertex normals of meshes by uid, kept while their ``"geometry"`` slot (points
    and triangles) keeps its version.

    A moving mesh gets new points every frame, so its normals are computed again;
    the :class:`.NormalAccumulator` of its triangles (which vertices share a point,
    which triangles each vertex is a corner of) is kept, so that costs one matrix
    product.
    """

    def __init__(self) -> None:
        self._current: dict[int, tuple[int, np.ndarray]] = {}
        self._previous: dict[int, tuple[int, np.ndarray]] = {}
        self._accumulators: dict[int, NormalAccumulator] = {}

    def next_frame(self) -> None:
        self._previous, self._current = self._current, {}
        live = self._previous.keys()
        self._accumulators = {
            uid: accumulator
            for uid, accumulator in self._accumulators.items()
            if uid in live
        }

    def get(self, node: RNode, mesh: RMesh) -> np.ndarray:
        key = node.versions.get("geometry", 0)
        entry = self._current.get(node.uid) or self._previous.get(node.uid)
        if entry is None or entry[0] != key:
            entry = (key, self._accumulator(node.uid, mesh).normals(mesh.points))
        self._current[node.uid] = entry
        return entry[1]

    def _accumulator(self, uid: int, mesh: RMesh) -> NormalAccumulator:
        kept = self._accumulators.get(uid)
        if kept is not None and kept.fits(mesh.points, mesh.triangles):
            return kept
        accumulator = NormalAccumulator(mesh.points, mesh.triangles)
        self._accumulators[uid] = accumulator
        return accumulator


class _TextureCache:
    """Image textures by uid, kept while the image is drawn in consecutive frames."""

    def __init__(self, renderer: OpenGLRenderer) -> None:
        self._renderer = renderer
        self._current: dict[int, tuple[int, moderngl.Texture]] = {}
        self._previous: dict[int, tuple[int, moderngl.Texture]] = {}
        self._registered = False

    def next_frame(self) -> None:
        for _, texture in self._previous.values():
            texture.release()
        self._previous, self._current = self._current, {}

    def get(
        self, node: RNode, slot: str, pixels: np.ndarray, nearest: bool = False
    ) -> moderngl.Texture | None:
        """The texture of ``pixels``, the value of the node's slot ``slot``; made
        anew if the node has no key or the slot changed.

        With ``nearest``, pixels are not smoothed when scaled.
        """
        version = node.versions.get(slot, 0)
        entry = self._current.get(node.uid) or self._previous.pop(node.uid, None)
        if entry is not None and entry[0] == version and node.key is not None:
            self._current[node.uid] = entry
            return entry[1]
        if entry is not None:
            entry[1].release()
        pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4 or 0 in pixels.shape:
            return None
        height, width = pixels.shape[:2]
        if not self._registered:
            # Textures belong to the renderer's context and go with it.
            self._renderer._resources.callback(self.release)
            self._registered = True
        texture = self._renderer.context.texture((width, height), 4, pixels.tobytes())
        if nearest:
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            texture.build_mipmaps()
            texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.repeat_x = texture.repeat_y = False
        self._current[node.uid] = (version, texture)
        return texture

    def release(self) -> None:
        for _, texture in (*self._current.values(), *self._previous.values()):
            if not isinstance(texture.mglo, moderngl.InvalidObject):
                texture.release()
        self._current.clear()
        self._previous.clear()
        self._registered = False
