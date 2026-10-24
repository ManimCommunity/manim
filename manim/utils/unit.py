"""Implement the Unit class."""

from __future__ import annotations

import numpy as np

from .. import config, constants
from ..typing import Vector3D

__all__ = ["Pixels", "Degrees", "Munits", "Percent"]


class _PixelUnits:
    def __mul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width

    def __rmul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width


class Percent:
    """A unit representing a percentage of the frame's width or height.

    Parameters
    ----------
    axis
        The axis whose frame length is used. Both directions of an axis are
        accepted, so ``RIGHT``, ``LEFT`` and ``X_AXIS`` refer to the frame
        width, and ``UP``, ``DOWN`` and ``Y_AXIS`` refer to the frame height.

    Raises
    ------
    NotImplementedError
        If ``axis`` is along the Z axis, whose length is undefined.
    ValueError
        If ``axis`` is not along the X, Y or Z axis.
    """

    def __init__(self, axis: Vector3D) -> None:
        # Compare the absolute value so that negative directions such as
        # LEFT and DOWN map to the same frame length as RIGHT and UP.
        direction = np.abs(np.asarray(axis, dtype=float))
        if np.array_equal(direction, constants.X_AXIS):
            self.length = config.frame_width
        elif np.array_equal(direction, constants.Y_AXIS):
            self.length = config.frame_height
        elif np.array_equal(direction, constants.Z_AXIS):
            raise NotImplementedError("length of Z axis is undefined")
        else:
            raise ValueError(
                f"Percent axis must be along the X, Y or Z axis, got {axis!r}"
            )

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
Degrees = constants.PI / 180
Munits = 1
