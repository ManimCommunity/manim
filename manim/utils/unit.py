"""Implement the Unit class."""

import numpy as np
from numpy import ndarray

from .. import config, constants

__all__ = ["Pixels", "Degrees", "Munits", "Percent"]


class _PixelUnits:
    def __mul__(self, val):
        return val * config.frame_width / config.pixel_width

    def __rmul__(self, val: int) -> float:
        return val * config.frame_width / config.pixel_width


class Percent:
    def __init__(self, axis: ndarray) -> None:
        if np.array_equal(axis, constants.X_AXIS):
            self.length = config.frame_width
        if np.array_equal(axis, constants.Y_AXIS):
            self.length = config.frame_height
        if np.array_equal(axis, constants.Z_AXIS):
            raise NotImplementedError("length of Z axis is undefined")

    def __mul__(self, val):
        return val / 100 * self.length

    def __rmul__(self, val: int) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
Degrees = constants.PI / 180
Munits: int = 1
