"""What renderers draw: immutable snapshots of mobjects.

A renderer draws a list of :class:`RNode` snapshots instead of reading mobjects. Each
node holds one payload with the values to draw (a path, a point cloud, an image, the
view of another camera), its place in the drawing order, and what a renderer needs to
reuse work from earlier frames:

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
(see :mod:`manim.mobject.fields`). The module imports no mobject module; snapshots
are made by :mod:`manim.renderer.extract`.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TYPE_CHECKING, NamedTuple

from manim.material import Pbr, Unlit

if TYPE_CHECKING:
    from PIL.Image import Image, Resampling

    from manim.constants import CapStyleType, LineJointType
    from manim.typing import (
        FloatRGBA,
        FloatRGBA_Array,
        PixelArray,
        Point3D,
        Point3D_Array,
    )

__all__ = [
    "LegacyShaded",
    "Material",
    "Payload",
    "RCameraView",
    "RImage",
    "RLinearGradient",
    "RNode",
    "RPaint",
    "RPath",
    "RPoints",
    "RSolid",
    "Stroke",
]


# Paints: how an area or a line is colored.


class RSolid(NamedTuple):
    """One color."""

    rgba: FloatRGBA


class RLinearGradient(NamedTuple):
    """Colors evenly spaced along the line from ``start`` to ``end``.

    The points are in scene coordinates. Beyond its ends, the gradient keeps the
    color of the nearer end.
    """

    start: Point3D
    end: Point3D
    rgbas: FloatRGBA_Array


RPaint = RSolid | RLinearGradient


class LegacyShaded(NamedTuple):
    """The shading of mobjects with ``shade_in_3d``: the stop colors of the path's
    paint, darkened or lightened towards the camera's light at the path's first and
    last corner, as :func:`~.get_shaded_rgb` does.

    A renderer whose camera has no light draws the paint as it is. This material is
    provisional; it goes when vector mobjects get lit materials.
    """


Material = Unlit | Pbr | LegacyShaded


class Stroke(NamedTuple):
    """How a path's outline is drawn."""

    paint: RPaint
    #: Line width in the units of :attr:`.VMobject.stroke_width`.
    width: float
    joint: LineJointType
    cap: CapStyleType


# Payloads: what is drawn for one mobject.


class RPath(NamedTuple):
    """A vectorized path (:class:`~.VMobject`).

    Slots of :attr:`RNode.versions`:

    - ``"geometry"``: ``points``, ``tolerance``
    - ``"fill"``: ``fill``, including the points a gradient's ends are taken from
    - ``"stroke"``, ``"background_stroke"``: ``stroke``, ``background_stroke``
      (paint as for the fill, width, joint and cap)
    - ``"material"``: ``material``
    - ``"depth_reference"``: ``depth_reference``
    - ``"mask"``: ``mask``
    """

    points: Point3D_Array
    fill: RPaint
    stroke: Stroke
    #: Drawn below the fill; ``None`` if it has no width.
    background_stroke: Stroke | None
    material: Material
    #: The point by which a camera with perspective orders paths by depth, or
    #: ``None`` if the path is drawn in scene order.
    depth_reference: Point3D | None
    #: Distance below which two points count as equal (closing subpaths).
    tolerance: float
    #: The image the path's colors are multiplied with, fixed to the frame (see
    #: :meth:`.VMobject.color_using_background_image`); provisional.
    mask: Image | str | None


class RPoints(NamedTuple):
    """A point cloud (:class:`~.PMobject`).

    Slots: ``"geometry"`` (``points``), ``"colors"`` (``rgbas``, ``stroke_width``).
    """

    points: Point3D_Array
    rgbas: FloatRGBA_Array
    stroke_width: float


class RImage(NamedTuple):
    """A raster image (:class:`~.AbstractImageMobject`), placed by its corner points.

    Slots: ``"geometry"`` (``points``), ``"pixels"`` (``pixel_array``,
    ``resampling_algorithm``).
    """

    pixel_array: PixelArray
    points: Point3D_Array
    resampling_algorithm: Resampling


class RCameraView(NamedTuple):
    """Where to show the view of another camera (:class:`~.ImageMobjectFromCamera`).

    The renderer supplies the pixels, looked up by the node's ``uid``. Nodes of camera
    views have no key: the view can change without the mobject changing.
    """

    points: Point3D_Array
    resampling_algorithm: Resampling
    #: ``id()`` of the camera whose view is shown; copies of a view share it.
    camera_id: int


Payload = RPath | RPoints | RImage | RCameraView


class RNode(NamedTuple):
    """What a renderer draws for one mobject."""

    #: Identifies the mobject within the process (see :attr:`.Mobject.uid`).
    uid: int
    #: Equal keys: the same drawing from the same view; ``None``: draw anew. See the
    #: module documentation.
    key: Hashable | None
    #: A version per slot of the payload; see the module documentation.
    versions: Mapping[str, int]
    z_index: float
    payload: Payload
    #: Drawn as if the camera were at its default position (see
    #: :meth:`.Mobject.fix_in_frame`).
    fixed_in_frame: bool
    #: The point the mobject keeps facing the camera from, or ``None`` (see
    #: :meth:`.Mobject.fix_orientation`).
    fixed_orientation_center: Point3D | None
