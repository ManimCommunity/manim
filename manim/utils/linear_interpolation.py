# LERP (Linear intERPolation) variants
from __future__ import annotations

__all__ = [
    "interpolate",
    "integer_interpolate",
    "mid",
    "inverse_interpolate",
    "match_interpolate",
]

import numpy as np


def interpolate(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start + alpha * (end - start)


# TODO: fix typing - it should be always tuple[int, float],
# but mypy complains because start and end are typed as floats, not ints.
# I'm not so sure about simply using int(end - 1) and int(start), won't
# it break anything?
def integer_interpolate(
    start: float, end: float, alpha: float
) -> tuple[int, float] | tuple[float, float]:
    """
    Alpha is a float between 0 and 1.  This returns
    an integer between start and end (inclusive) representing
    appropriate interpolation between them, along with a
    "residue" representing a new proportion between the
    returned integer and the next one of the
    list.

    For example, if start=0, end=10, alpha=0.46, This
    would return (4, 0.6).
    """
    if alpha >= 1:
        return (end - 1, 1.0)
    if alpha <= 0:
        return (start, 0)
    value = int(interpolate(start, end, alpha))
    residue = ((end - start) * alpha) % 1
    return (value, residue)


def mid(start: float, end: float) -> float:
    return (start + end) / 2.0


def inverse_interpolate(start: float, end: float, value: float) -> np.ndarray:
    return np.true_divide(value - start, end - start)


def match_interpolate(
    new_start: float,
    new_end: float,
    old_start: float,
    old_end: float,
    old_value: float,
) -> np.ndarray:
    return interpolate(
        new_start,
        new_end,
        inverse_interpolate(old_start, old_end, old_value),
    )
