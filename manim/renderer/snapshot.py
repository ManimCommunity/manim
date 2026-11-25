"""What renderers draw: immutable snapshots of mobjects.

A renderer draws a list of :class:`RNode` snapshots instead of reading mobjects. Each
node holds one payload with the values to draw (a path, a point cloud, an image, the
view of another camera), how it is placed in the view, and what a renderer needs to
reuse work from earlier frames. The list is in drawing order (family order, sorted by
``z_index``; see :func:`~.extract`).

The ``R`` in the names stands for *render snapshot*: each ``R`` type holds the values
of one mobject or style type, resolved for drawing (``RPath`` for a
:class:`~.VMobject`, ``RSolidColor`` for a :class:`~.SolidColor` paint, and so on;
each class says which).
Values are resolved where a renderer would otherwise need the mobject: a gradient's
ends are points in the scene, not a direction relative to the mobject.

Each node also says how it is placed (:attr:`RNode.placement`):

- :class:`InScene`: in the scene, drawn through the camera.
- :class:`FixedInFrame`: drawn as if the camera were at its default position (see
  :meth:`.Mobject.fix_in_frame`).
- :class:`FacingCamera`: moves with a point of the scene, but is not rotated with it
  (see :meth:`.Mobject.fix_orientation`).

For reuse:

- :attr:`RNode.key`: equal keys mean the same drawing from the same view. A renderer
  may reuse whatever it made for a node with an equal key. ``None`` means that the
  node may change without its mobject's fields changing: such a node is drawn anew
  every time, and ``None`` never equals another key.
- :attr:`RNode.versions`: one version per part of the payload ("slot", listed with
  each payload type), for renderers that keep parts of a node (such as uploaded
  geometry) separately. An equal version of a slot means an interchangeable value;
  a value may get a new version without changing.

A renderer may also ignore both and draw everything every frame.

Payload arrays are the mobjects' field values: read-only, and shared, not copied
(see :mod:`manim.utils.versioned_fields`). The module imports no mobject module; snapshots
are made by :mod:`manim.renderer.extract`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

from manim.appearance.material import Material

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Resampling

    from manim.constants import CapStyleType, LineJointType
    from manim.typing import (
        FloatRGBA,
        FloatRGBA_Array,
        PixelArray,
        Point3D,
        Point3D_Array,
    )

__all__ = [
    "CameraId",
    "FacingCamera",
    "FixedInFrame",
    "Payload",
    "Placement",
    "RCameraView",
    "RColorField",
    "RImage",
    "RLinearGradient",
    "RMesh",
    "RNode",
    "RPaint",
    "RPath",
    "RPoints",
    "RSolidColor",
    "STROKE_WIDTH_UNIT",
    "RStroke",
]


# Paints: how an area or a line is colored.


#: Identifies a mobject within the process (:attr:`.Mobject.uid`); copies get new ones.
MobjectUid: TypeAlias = int
#: Identifies the camera a camera view shows (its ``id()``); copies of a view share it.
CameraId: TypeAlias = int


class RSolidColor(NamedTuple):
    """One color: the snapshot of a :class:`~.SolidColor` paint."""

    rgba: FloatRGBA


class RLinearGradient(NamedTuple):
    """Colors evenly spaced along the line from ``start`` to ``end``: the snapshot of
    a mobject's list of fill or stroke colors, its gradient.

    The points are in scene coordinates. Beyond its ends, the gradient keeps the
    color of the nearer end.
    """

    start: Point3D
    end: Point3D
    rgbas: FloatRGBA_Array


class RColorField(NamedTuple):
    """Colours in scene coordinates (a :class:`~.ColorField`), sampled where the
    path is drawn.

    ``grid``: ``(rows, columns, 4)`` straight-alpha sRGB floats, row 0 at the top;
    ``corners``: the upper left, upper right and lower left scene point of the
    grid's outer edges; ``opacity`` multiplies the alpha. The colour at a point
    is :meth:`.ColorField.sample`'s. ``grid_id`` identifies the grid's storage
    only (equal ids, equal colours): one texture can serve many paints.
    """

    grid: np.ndarray
    corners: Point3D_Array
    opacity: float
    grid_id: int


RPaint = RSolidColor | RLinearGradient | RColorField


class RStroke(NamedTuple):
    """How a path's outline is drawn."""

    paint: RPaint
    #: Line width in the units of :attr:`.VMobject.stroke_width`.
    width: float
    joint: LineJointType
    cap: CapStyleType


# Payloads: what is drawn for one mobject.


class RPath(NamedTuple):
    """A vectorized path: the snapshot of a :class:`~.VMobject`.

    Slots of :attr:`RNode.versions`:

    - ``"geometry"``: ``points``, ``tolerance``
    - ``"fill"``: ``fill``, including the points a gradient's ends are taken from
    - ``"stroke"``, ``"background_stroke"``: ``stroke``, ``background_stroke``
      (paint as for the fill, width, joint and cap)
    - ``"material"``: ``material``
    - ``"depth_reference"``: ``depth_reference``
    """

    points: Point3D_Array
    fill: RPaint
    stroke: RStroke
    #: Drawn below the fill; ``None`` if it has no width.
    background_stroke: RStroke | None
    material: Material
    #: The point by which a camera with perspective orders paths by depth, or
    #: ``None`` if the path is drawn in scene order.
    depth_reference: Point3D | None
    #: Distance below which two points count as equal (closing subpaths).
    tolerance: float


class RPoints(NamedTuple):
    """A point cloud: the snapshot of a :class:`~.PMobject`.

    Slots: ``"geometry"`` (``points``), ``"colors"`` (``rgbas``, ``stroke_width``).
    """

    points: Point3D_Array
    rgbas: FloatRGBA_Array
    stroke_width: float


class RImage(NamedTuple):
    """A raster image, placed by its corner points: the snapshot of an
    :class:`~.AbstractImageMobject`.

    Slots: ``"geometry"`` (``points``), ``"pixels"`` (``pixel_array``,
    ``resampling_algorithm``).
    """

    pixel_array: PixelArray
    points: Point3D_Array
    resampling_algorithm: Resampling


class RCameraView(NamedTuple):
    """Where to show the view of another camera: the snapshot of an
    :class:`~.ImageMobjectFromCamera`.

    The renderer supplies the pixels, looked up by the node's ``uid``. Nodes of camera
    views have no key: the view can change without the mobject changing.
    """

    points: Point3D_Array
    resampling_algorithm: Resampling
    #: ``id()`` of the camera whose view is shown; copies of a view share it.
    camera_id: CameraId


class RMesh(NamedTuple):
    """A triangle mesh (:class:`~.MeshMobject`).

    Slots: ``"geometry"`` (``points``, ``triangles``, ``faces``), ``"rgbas"``,
    ``"uvs"``, ``"texture"``, ``"material"``, ``"stroke"`` (``stroke_rgbas``,
    ``stroke_width``).
    """

    points: Point3D_Array
    #: Vertex indices, three per triangle.
    triangles: np.ndarray
    #: One color for the whole mesh, or one per vertex.
    rgbas: FloatRGBA_Array
    #: Texture coordinates per vertex; empty without a texture.
    uvs: np.ndarray
    #: RGBA pixels; empty without a texture.
    texture: PixelArray
    material: Material
    #: Whether the mesh's fragments hide each other and what is behind them.
    depth_test: bool
    #: Polygons as vertex indices, in order around each (see
    #: :attr:`.MeshMobject.faces`); empty: the triangles.
    faces: np.ndarray
    #: Outline colors of the faces: one, or one per face.
    stroke_rgbas: FloatRGBA_Array
    #: Outline width, in the units of :attr:`.VMobject.stroke_width`; 0 for none.
    stroke_width: float


Payload = RPath | RPoints | RImage | RCameraView | RMesh


# Placements: how a node is placed in the view.


class InScene(NamedTuple):
    """In the scene, drawn through the camera: how most nodes are placed."""


class FixedInFrame(NamedTuple):
    """Drawn as if the camera were at its default position and orientation (see
    :meth:`.Mobject.fix_in_frame`). Takes precedence over a fixed orientation.
    """


class FacingCamera(NamedTuple):
    """Moved with the projection of ``center``, but not rotated with the scene (see
    :meth:`.Mobject.fix_orientation`).
    """

    center: Point3D


Placement = InScene | FixedInFrame | FacingCamera


class RNode(NamedTuple):
    """What a renderer draws for one mobject: the snapshot of a :class:`~.Mobject`."""

    #: Identifies the mobject within the process (see :attr:`.Mobject.uid`).
    uid: MobjectUid
    #: Equal keys: the same drawing from the same view; ``None``: draw anew. See the
    #: module documentation.
    key: Hashable | None
    #: A version per slot of the payload; see the module documentation.
    versions: Mapping[str, int]
    payload: Payload
    #: How the node is placed in the view. See the module documentation.
    placement: Placement

    @property
    def is_fixed_in_frame(self) -> bool:
        """Whether the node is drawn as if the camera were at its default position."""
        return type(self.placement) is FixedInFrame
