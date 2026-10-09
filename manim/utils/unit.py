"""Units for converting between Manim's internal coordinate system and physical measurements.

Manim scenes are laid out in an internal coordinate system whose base unit is
the *manim unit* (:obj:`.Munits`).  The helpers in this module make it easy to
express distances and angles in more familiar units (pixels, degrees,
percentages) and convert them into the corresponding manim units, so that they
can be used wherever a length or an angle is expected.

The following converters are available:

- :obj:`.Pixels` -- convert a number of screen pixels into manim units.
- :obj:`.Degrees` -- convert an angle given in degrees into radians.
- :obj:`.Munits` -- the base unit itself; ``x * Munits`` is simply ``x``.
- :obj:`.Percent` -- convert a percentage of an axis' length into manim units.

Because every converter implements the multiplication protocol, you simply
multiply a plain number by the converter you need.

Examples
--------
.. code-block:: pycon

    >>> from manim import unit
    >>> # 50 pixels, expressed in manim units
    >>> 50 * unit.Pixels
    0.37037037037037035
    >>> # 90 degrees, expressed in radians (manim units)
    >>> 90 * unit.Degrees
    1.5707963267948966
    >>> # 10% of the width of the x-axis, expressed in manim units
    >>> from manim import X_AXIS
    >>> unit.Percent(X_AXIS) * 10
    1.4222222222222223

"""

from __future__ import annotations

import numpy as np

from .. import config, constants
from ..typing import Vector3D

__all__ = ["Pixels", "Degrees", "Munits", "Percent"]


class _PixelUnits:
    """Convert a number of screen pixels into manim units.

    Multiplying a number by :obj:`~manim.utils.unit.Pixels` returns the
    equivalent length in manim units.  The conversion is based on the configured
    frame width and pixel width (see
    :attr:`~manim.utils.cfg.config.frame_width` and
    :attr:`~manim.utils.cfg.config.pixel_width`), so the result depends on the
    active camera configuration.

    Examples
    --------
    .. code-block:: pycon

        >>> from manim import unit
        >>> 50 * unit.Pixels
        0.37037037037037035

    """

    def __mul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width

    def __rmul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width


class Percent:
    """Convert a percentage of an axis' length into manim units.

    ``Percent(axis)`` represents one percent of the length of ``axis``.  Multiply
    it by a number to obtain that percentage of the axis expressed in manim
    units.  The supported axes are :data:`~manim.constants.X_AXIS` and
    :data:`~manim.constants.Y_AXIS`; the length of
    :data:`~manim.constants.Z_AXIS` is undefined and raises
    :exc:`NotImplementedError`.

    Parameters
    ----------
    axis : :class:`~numpy.ndarray`
        One of :data:`~manim.constants.X_AXIS`, :data:`~manim.constants.Y_AXIS`
        or :data:`~manim.constants.Z_AXIS`.

    Examples
    --------
    .. code-block:: pycon

        >>> from manim import unit, X_AXIS
        >>> unit.Percent(X_AXIS) * 10
        1.4222222222222223

    """

    def __init__(self, axis: Vector3D) -> None:
        if np.array_equal(axis, constants.X_AXIS):
            self.length = config.frame_width
        if np.array_equal(axis, constants.Y_AXIS):
            self.length = config.frame_height
        if np.array_equal(axis, constants.Z_AXIS):
            raise NotImplementedError("length of Z axis is undefined")

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
"""A converter that turns a number of screen pixels into manim units.

See :class:`~manim.utils.unit._PixelUnits` for details and examples.
"""
# One degree expressed in radians (manim units). Multiply an angle given in
# degrees by :obj:`~manim.utils.unit.Degrees` to obtain its value in radians,
# which is what Manim expects for angles.
# Example: ``90 * Degrees`` equals ``1.5707963267948966`` (pi / 2).
Degrees = constants.PI / 180
# The base manim unit. ``x * Munits`` is simply ``x``; it is provided so that
# code can be explicit about which unit a quantity is expressed in.
Munits = 1
