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
    def __init__(self, axis: Vector3D) -> None:
        if np.array_equal(axis, constants.X_AXIS) or np.array_equal(
            axis, -constants.X_AXIS
        ):
            self.length = config.frame_width
        elif np.array_equal(axis, constants.Y_AXIS) or np.array_equal(
            axis, -constants.Y_AXIS
        ):
            self.length = config.frame_height
        else:
            raise ValueError("Percent axis must be X_AXIS or Y_AXIS.")

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
Degrees = constants.PI / 180
Munits = 1
