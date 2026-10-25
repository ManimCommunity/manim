"""Implement the Unit class."""

from __future__ import annotations

import numpy as np

from .. import config, constants
from ..typing import Vector3DLike

__all__ = ["Pixels", "Degrees", "Munits", "Percent"]


class _PixelUnits:
    def __mul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width

    def __rmul__(self, val: float) -> float:
        return val * config.frame_width / config.pixel_width


class Percent:
    """A percentage of the frame size, measured along a direction.

    The length used is the extent of the frame through its centre along
    ``axis``. For :data:`~.RIGHT` and :data:`~.LEFT` this is the frame width,
    and for :data:`~.UP` and :data:`~.DOWN` it is the frame height. Only the
    direction of ``axis`` matters, not its length, and its z component is
    ignored because the frame has no depth.

    Parameters
    ----------
    axis
        The direction along which the percentage is measured.

    Raises
    ------
    NotImplementedError
        If ``axis`` points along the z axis only, since the frame has no depth.
    ValueError
        If ``axis`` is the zero vector or contains non-finite values.

    Examples
    --------
    .. code-block:: pycon

        >>> from manim import LEFT, UP
        >>> float(50 * Percent(LEFT)) == config.frame_width / 2
        True
        >>> float(25 * Percent(UP)) == config.frame_height / 4
        True
    """

    def __init__(self, axis: Vector3DLike) -> None:
        axis = np.asarray(axis, dtype=float)
        if not np.all(np.isfinite(axis)):
            raise ValueError("axis must only contain finite values")
        planar_norm = np.hypot(axis[0], axis[1])
        if planar_norm == 0:
            if np.any(axis[2:]):
                raise NotImplementedError("length of Z axis is undefined")
            raise ValueError("axis must be a non-zero vector")
        self.direction = axis[:2] / planar_norm

    @property
    def length(self) -> float:
        """The extent of the frame through its centre along :attr:`direction`."""
        dx, dy = np.abs(self.direction)
        candidates = []
        if dx > 0:
            candidates.append(config.frame_width / dx)
        if dy > 0:
            candidates.append(config.frame_height / dy)
        return float(min(candidates))

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
Degrees = constants.PI / 180
Munits = 1
