"""How the paint of a path snapshot becomes colors, for every renderer.

Renderers draw a paint as stop colors, evenly spaced along a line between two points.
:func:`paint_colors` gives both, with the shading of a :class:`.LegacyShaded` material
applied by the camera.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from manim.renderer.snapshot import LegacyShaded, RSolidColor
from manim.utils._path_points import end_corner, start_corner

if TYPE_CHECKING:
    from manim.renderer.snapshot import RPaint, RPath
    from manim.typing import FloatRGBA_Array, Point3D_Array

__all__ = ["paint_colors", "shows"]

#: Shades the stop colors of a path's paint (``Camera._shaded_rgbas``).
Shade = Callable[["RPath", "FloatRGBA_Array"], "FloatRGBA_Array"]


def shows(paint: RPaint) -> bool:
    """Whether ``paint`` is visible anywhere (not fully transparent)."""
    if isinstance(paint, RSolidColor):
        return bool(paint.rgba[3] > 0)
    return bool(paint.rgbas[:, 3].max() > 0)


def paint_colors(
    paint: RPaint, path: RPath, shade: Shade
) -> tuple[FloatRGBA_Array, Point3D_Array | None]:
    """The stop colors of ``paint`` on ``path`` and the start and end point of the
    line they are spaced along; ``None`` for one color.
    """
    if isinstance(paint, RSolidColor):
        rgbas = np.asarray(paint.rgba).reshape(1, 4)
        if type(path.material) is not LegacyShaded:
            return rgbas, None
        rgbas = shade(path, rgbas)
        if len(rgbas) == 1:
            return rgbas, None
        # Only shading turns a solid paint into a gradient: between the path's
        # first and last corner, as the shading itself.
        return rgbas, np.array([start_corner(path.points), end_corner(path.points)])
    rgbas = shade(path, paint.rgbas)
    return rgbas, np.array([paint.start, paint.end])
