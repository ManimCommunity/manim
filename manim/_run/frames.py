"""How many frames a span of scene time takes."""

from __future__ import annotations

import math

__all__: list[str] = []

# Durations computed from frame counts, such as 0.28 * 25, land a little off
# whole numbers; they must not get an extra frame.
_TOLERANCE = 1e-9


def frame_count(duration: float, frame_rate: float) -> int:
    """The number of frames shown during ``duration`` seconds.

    A frame shows the scene at its start time, so these are the frames whose start
    times ``i / frame_rate`` lie before ``duration``.
    """
    return max(0, math.ceil(duration * frame_rate - _TOLERANCE))
