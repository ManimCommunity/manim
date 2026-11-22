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
from typing import TYPE_CHECKING, NamedTuple, cast

import moderngl
import numpy as np

from manim.renderer._paint import paint_colors, shows
from manim.renderer.snapshot import LegacyShaded, RNode, RPath, RSolid

from .shader import Shader
from .winding_fill import WindingFill

if TYPE_CHECKING:
    from .renderer import OpenGLRenderer

__all__ = ["SnapshotDrawing", "cubics_to_quadratics"]

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
    """Colours at ``points`` with Cairo's semantics.

    One colour fills everything; several are stops of a linear gradient, evenly
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
    def of(cls, renderer: OpenGLRenderer) -> _Screen:
        view = renderer.camera_view
        pixel_shape = renderer.get_pixel_shape()
        assert pixel_shape is not None
        size = np.array(pixel_shape, dtype=float)
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

    def release(self) -> None:
        self._fill.release()

    def draw(self, nodes: Iterable[RNode]) -> None:
        self._geometry.next_frame()
        paths = [node for node in nodes if isinstance(node.payload, RPath)]
        if not paths:
            return
        tolerance = self._tolerance()
        geometry = self._geometry.get(paths, tolerance)
        screen = _Screen.of(self._renderer)
        for group in _groups(paths, geometry, screen):
            self._draw_group(group, screen)

    def _tolerance(self) -> float:
        """The conversion tolerance in scene units, rounded down to a power of two.

        Rounding keeps cached geometry valid while the camera zooms a little.
        """
        pixel_shape = self._renderer.get_pixel_shape()
        assert pixel_shape is not None
        frame_width = self._renderer.camera_view.frame_shape[0]
        tolerance = TOLERANCE_PIXELS * frame_width / pixel_shape[0]
        return 2.0 ** math.floor(math.log2(tolerance))

    def _draw_group(self, group: _Group, screen: _Screen) -> None:
        if any(_background_width(n.payload) > 0 for n in group.nodes):  # type: ignore[arg-type]
            self._draw_strokes(group, background=True)
        if group.filled:
            self._fill.draw(group.nodes, group.geometry, screen.scissor(group.box))
        if any(n.payload.stroke.width > 0 for n in group.nodes):  # type: ignore[union-attr]
            self._draw_strokes(group, background=False)

    def _draw_strokes(self, group: _Group, background: bool) -> None:
        camera = self._renderer.camera
        geometry, widths, colours = [], [], []
        for node, geo in zip(group.nodes, group.geometry, strict=True):
            path: RPath = node.payload  # type: ignore[assignment]
            stroke = path.background_stroke if background else path.stroke
            if stroke is None or stroke.width == 0 or len(geo.quads) == 0:
                continue
            geometry.append(geo)
            widths.append(stroke.width)
            colours.append(paint_colors(stroke.paint, path, camera._shaded_rgbas))
        if not geometry:
            return
        vertices = _stroke_vertices(geometry, widths, colours)
        renderer = self._renderer
        context = renderer.context
        shader = Shader(context, name="quadratic_bezier_stroke")
        for name, value in renderer.perspective_uniforms.items():
            shader.set_uniform(name, value)
        first: RPath = group.nodes[0].payload  # type: ignore[assignment]
        shader.set_uniform("joint_type", float(first.stroke.joint.value))
        shader.set_uniform("flat_stroke", 0.0)
        for name in ("gloss", "shadow", "is_fixed_in_frame", "is_fixed_orientation"):
            shader.set_uniform(name, 0.0)
        buffer = context.buffer(vertices.tobytes())
        try:
            array = context.vertex_array(
                shader.shader_program,
                [(buffer, _STROKE_FORMAT, *(field[0] for field in _STROKE_DTYPE))],
            )
            try:
                context.disable(moderngl.DEPTH_TEST)
                context.enable(moderngl.BLEND)
                context.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
                renderer.frame_buffer_object.use()
                array.render(moderngl.TRIANGLES)
            finally:
                array.release()
        finally:
            buffer.release()


def _stroke_vertices(
    geometry: list[_Geometry],
    widths: list[float],
    colours: list[tuple[np.ndarray, np.ndarray | None]],
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
    vertices["color"] = _colours(colours, points, 3 * counts)
    return vertices


def _concatenate(
    geometry: Sequence[_Geometry],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All pieces of several paths, pieces per path and each path's first piece."""
    counts = np.array([len(g.quads) for g in geometry])
    quads = np.concatenate([g.quads for g in geometry])
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    return quads, counts, offsets


def _colours(
    colours: list[tuple[np.ndarray, np.ndarray | None]],
    points: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Vertex colours of several paths; ``counts`` vertices per path."""
    first = np.array([rgbas[0] for rgbas, _ in colours])
    result = np.repeat(first, counts, axis=0)
    ends = np.cumsum(counts)
    for (rgbas, gradient), end, count in zip(colours, ends, counts, strict=True):
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
    they are filled, otherwise their stroke pieces). Paths of one colour without
    strokes, such as the glyphs of a text, may also overlap each other: their union
    is coloured once. For opaque paths that gives the same pixels; for translucent
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

    groups: list[_Group] = []
    regions = _Boxes()
    colour: tuple[float, ...] | None = None
    joint = None
    for index, (node, geo, path) in enumerate(zip(nodes, geometry, paths, strict=True)):
        box = boxes[index]
        single = (
            tuple(path.fill.rgba)
            if isinstance(path.fill, RSolid)
            and type(path.material) is not LegacyShaded
            and path.stroke.width == 0
            else None
        )
        joins = bool(groups) and path.stroke.joint == joint
        if joins and filled[index]:
            same_colour = single is not None and single == colour
            joins = same_colour or not (
                _overlap(groups[-1].box, box) and regions.overlaps(box)
            )
        if filled[index] or len(geo.quads) == 0:
            region = box[None, :]
        else:
            region = piece_boxes[piece_offsets[index] : piece_offsets[index + 1]]
        if joins:
            group = groups[-1]
            group.nodes.append(node)
            group.geometry.append(geo)
            if filled[index] and not group.filled:
                colour = single
            groups[-1] = group._replace(
                box=np.r_[
                    np.minimum(group.box[:2], box[:2]),
                    np.maximum(group.box[2:], box[2:]),
                ],
                filled=group.filled or bool(filled[index]),
            )
        else:
            groups.append(_Group([node], [geo], bool(filled[index]), box))
            regions = _Boxes()
            colour = single
            joint = path.stroke.joint
        regions.add(region)
    return groups


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1])


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
