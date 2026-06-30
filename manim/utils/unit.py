"""Unit conversion helpers for Manim coordinates.

Manim scenes use abstract *Munits* (Manim units) for positioning mobjects.
These helpers convert pixels, degrees, and screen percentages into Munits.

Examples
--------
.. code-block:: pycon

    >>> from manim import unit, X_AXIS
    >>> 50 * unit.Pixels  # 50 pixels -> Munits
    >>> 90 * unit.Degrees  # degrees -> radians
    >>> unit.Percent(X_AXIS) * 10  # 10% of frame width
"""

from __future__ import annotations

import numpy as np

from .. import config, constants
from ..typing import Vector3D

__all__ = ["Pixels", "Degrees", "Munits", "Percent"]


class _PixelUnits:
    """Convert pixel counts to Munits using the current frame width."""

    def __mul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width

    def __rmul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width


class Percent:
    """Convert a percentage of the frame width or height to Munits.

    Parameters
    ----------
    axis
        Either ``X_AXIS`` or ``Y_AXIS``. ``Z_AXIS`` is not supported.

    Examples
    --------
    .. code-block:: pycon

        >>> from manim import unit, X_AXIS, Y_AXIS
        >>> unit.Percent(X_AXIS) * 10  # 10% of frame width
        >>> unit.Percent(Y_AXIS) * 25  # 25% of frame height
    """

    def __init__(self, axis: Vector3D) -> None:
        if np.array_equal(axis, constants.X_AXIS):
            self.length = config.frame_width
        elif np.array_equal(axis, constants.Y_AXIS):
            self.length = config.frame_height
        elif np.array_equal(axis, constants.Z_AXIS):
            raise NotImplementedError("length of Z axis is undefined")
        else:
            raise ValueError("Percent axis must be X_AXIS or Y_AXIS.")

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
"""Convert pixel counts to Munits: ``50 * Pixels``."""

Degrees = constants.PI / 180
"""Convert degrees to radians: ``90 * Degrees``."""

Munits = 1
"""Identity unit for Manim coordinates."""
