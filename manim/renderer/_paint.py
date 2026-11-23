"""How the paint of a path snapshot becomes colors, for every renderer.

Renderers draw a paint as stop colors, evenly spaced along a line between two points.
:func:`paint_colors` gives both; :func:`lit_color` gives the one colour of a lit
(:class:`~.Pbr`) path's fill.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from manim.material import Pbr
from manim.renderer._lighting import Light, shade
from manim.renderer.protocol import CapabilityError
from manim.renderer.snapshot import RColorField, RSolidColor

if TYPE_CHECKING:
    from manim.renderer.protocol import LightState
    from manim.renderer.snapshot import RNode, RPaint, RPath
    from manim.typing import FloatRGBA_Array, Point3D, Point3D_Array

__all__ = ["lit_color", "paint_colors", "shows"]

#: Relative tolerance below which a path's points count as one plane.
_FLAT_TOLERANCE = 1e-9


def shows(paint: RPaint) -> bool:
    """Whether ``paint`` is visible anywhere (not fully transparent)."""
    if isinstance(paint, RSolidColor):
        return bool(paint.rgba[3] > 0)
    if isinstance(paint, RColorField):
        return paint.opacity > 0 and bool(np.asarray(paint.grid)[..., 3].max() > 0)
    return bool(paint.rgbas[:, 3].max() > 0)


def paint_colors(
    paint: RPaint, lit: FloatRGBA_Array | None = None
) -> tuple[FloatRGBA_Array, Point3D_Array | None]:
    """The stop colors of ``paint`` and the start and end point of the line they are
    spaced along; ``None`` for one color.

    ``lit`` is the colour of a lit path's fill (see :func:`lit_color`), which
    replaces the paint.
    """
    if lit is not None:
        return np.asarray(lit, dtype=float).reshape(1, 4), None
    if isinstance(paint, RColorField):
        # Renderers sample the field where they draw; this colour is not used.
        return np.ones((1, 4)), None
    if isinstance(paint, RSolidColor):
        return np.asarray(paint.rgba).reshape(1, 4), None
    return paint.rgbas, np.array([paint.start, paint.end])


def lit_color(
    node: RNode,
    eye: Point3D,
    lights: Sequence[LightState],
    ambient: Sequence[float],
) -> FloatRGBA_Array | None:
    """The one colour of a lit path's fill: :func:`.shade` at the center of its
    anchors, with the normal of their plane, two-sided. ``None`` for a path that is
    not lit (:class:`.Unlit`, fixed in the frame, or no lights).

    Raises
    ------
    CapabilityError
        For a lit path that is not flat or whose fill is not one colour.
    """
    path: RPath = node.payload  # type: ignore[assignment]
    if not isinstance(path.material, Pbr) or not lights or node.fixed_in_frame:
        return None
    points = np.asarray(path.points, dtype=float)
    if len(points) == 0 or not shows(path.fill):
        return None
    if not isinstance(path.fill, RSolidColor):
        raise CapabilityError(
            "A lit (Pbr) path is drawn in one colour: its fill must be a solid "
            "colour, not a gradient or a colour field."
        )
    anchors = np.concatenate([points[::4], points[3::4]])
    center = anchors.mean(axis=0)
    normal = _plane_normal(anchors - center)
    rgba = np.array(path.fill.rgba, dtype=float)
    rgba[:3] = shade(
        rgba[None, :3],
        normal[None],
        center[None],
        np.asarray(eye, dtype=float),
        [Light(light.position, light.rgb, light.intensity) for light in lights],
        ambient,
        path.material,
    )[0]
    return rgba


def _plane_normal(offsets: np.ndarray) -> np.ndarray:
    """The normal of the plane of points (given relative to their center).

    Raises
    ------
    CapabilityError
        If the points do not lie in one plane.
    """
    scale = max(float(np.abs(offsets).max()), 1.0)
    if np.ptp(offsets[:, 2]) <= _FLAT_TOLERANCE * scale:
        return np.array([0.0, 0.0, 1.0])
    _, singular, vt = np.linalg.svd(offsets, full_matrices=False)
    if singular[-1] > _FLAT_TOLERANCE * scale * max(len(offsets), 1) ** 0.5:
        raise CapabilityError(
            "A lit (Pbr) path is drawn in one colour and must be flat; this one is not."
        )
    return vt[-1]
