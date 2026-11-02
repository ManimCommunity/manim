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
    """A percentage of the frame width or height.

    Parameters
    ----------
    axis
        Which frame dimension to measure: ``X_AXIS`` for a percentage of
        ``config.frame_width``, or ``Y_AXIS`` for a percentage of
        ``config.frame_height``.

    Raises
    ------
    NotImplementedError
        If ``axis`` is ``Z_AXIS``.
    ValueError
        If ``axis`` is any other vector.

    Notes
    -----
    The length is based on ``config.frame_width`` and
    ``config.frame_height``, not on the current camera frame, so it may
    not match what is on screen after the camera is moved or zoomed.

    Examples
    --------
    >>> from manim import X_AXIS, config
    >>> from manim.utils.unit import Percent
    >>> 50 * Percent(X_AXIS) == config.frame_width / 2
    True
    """

    def __init__(self, axis: Vector3D) -> None:
        if np.array_equal(axis, constants.X_AXIS):
            self.length = config.frame_width
        elif np.array_equal(axis, constants.Y_AXIS):
            self.length = config.frame_height
        elif np.array_equal(axis, constants.Z_AXIS):
            raise NotImplementedError("length of Z axis is undefined")
        else:
            raise ValueError(
                "Percent only supports X_AXIS (frame width) and "
                f"Y_AXIS (frame height), got {np.asarray(axis).tolist()}."
            )

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
Degrees = constants.PI / 180
Munits = 1
