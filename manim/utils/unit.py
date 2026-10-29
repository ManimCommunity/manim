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
    """A percentage of the frame size along a given vector.

    Each component of ``axis`` is scaled by the matching frame dimension
    (the frame width for x, the frame height for y, and a depth of ``0``
    for z), and the length of the result is used. So :data:`~.RIGHT` and
    :data:`~.LEFT` give the frame width, :data:`~.UP` and :data:`~.DOWN`
    give the frame height, ``2 * RIGHT`` gives twice the frame width,
    and :data:`~.OUT` or the zero vector give ``0``.

    Parameters
    ----------
    axis
        The vector along which the percentage is measured.

    Raises
    ------
    ValueError
        If ``axis`` contains non-finite values.

    Examples
    --------
    .. code-block:: pycon

        >>> from manim import LEFT, UP
        >>> float(50 * Percent(LEFT)) == config.frame_width / 2
        True
        >>> float(25 * Percent(2 * UP)) == config.frame_height / 2
        True
    """

    def __init__(self, axis: Vector3DLike) -> None:
        axis = np.asarray(axis, dtype=float)
        if not np.all(np.isfinite(axis)):
            raise ValueError("axis must only contain finite values")
        self.axis = axis

    @property
    def length(self) -> float:
        """The length of :attr:`axis` with each component scaled by the frame size."""
        frame_size = np.array([config.frame_width, config.frame_height, 0.0])
        return float(np.linalg.norm(self.axis * frame_size[: len(self.axis)]))

    def __mul__(self, val: float) -> float:
        return val / 100 * self.length

    def __rmul__(self, val: float) -> float:
        return val / 100 * self.length


Pixels = _PixelUnits()
Degrees = constants.PI / 180
Munits = 1
