"""Paints: how an area or a line of a vectorized mobject is coloured.

A paint is an immutable value. :class:`Solid` is one colour; :class:`LinearGradient`
is evenly spaced colour stops along a direction, between the edges of the mobject's
own bounding box. Assign one to a mobject's ``fill``, ``stroke`` or
``background_stroke``, or let :meth:`.VMobject.set_fill` and
:meth:`.VMobject.set_stroke` make it::

    square.fill = LinearGradient([RED, BLUE], direction=RIGHT)
    square.set_fill([RED, BLUE], opacity=1)  # the same, with the default direction

Paints are compared by identity: two equal paints made separately are two values.
A gradient's end points are recomputed from the mobject's current points (its own
bounding box projected on ``direction``); :meth:`.VMobject.rotate` turns the
direction, but on a shape that is not square the end points do not turn rigidly
with it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from manim.constants import UL
from manim.utils.color import ManimColor, ParsableManimColor

if TYPE_CHECKING:
    from typing import Self

    from manim.typing import FloatRGBA_Array, Vector3D, Vector3DLike

__all__ = ["LinearGradient", "Paint", "Solid", "interpolate_paint"]


def _frozen(array: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.array(array, dtype=np.float64)
    if shape is not None:
        result = result.reshape(shape)
    result.flags.writeable = False
    return result


class Paint:
    """A colouring of an area or a line; immutable. See :mod:`manim.paint`."""

    __slots__ = ("_rgbas",)
    _rgbas: FloatRGBA_Array

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable.")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (Paint._from_rgbas, (self._rgbas, getattr(self, "_direction", None)))

    @property
    def rgbas(self) -> FloatRGBA_Array:
        """The colour stops as a read-only float array of RGBA rows (one row for a
        :class:`Solid`).
        """
        return self._rgbas

    @property
    def colors(self) -> list[ManimColor]:
        """The colours of the stops."""
        return [ManimColor(rgba[:3]) for rgba in self._rgbas]

    @property
    def opacities(self) -> np.ndarray:
        """The opacities of the stops (read-only)."""
        return self._rgbas[:, 3]

    def with_opacity(self, opacity: float | Sequence[float]) -> Paint:
        """The same colours with ``opacity`` (one for all stops, or one per stop)."""
        rgbas = self._rgbas.copy()
        opacities = np.asarray(opacity, dtype=float)
        if opacities.ndim and len(opacities) != len(rgbas):
            rgbas = _stretched(rgbas, max(len(rgbas), len(opacities)))
            opacities = _stretched(opacities, len(rgbas))
        rgbas[:, 3] = opacities
        return Paint._from_rgbas(rgbas, getattr(self, "_direction", None))

    def with_colors(self, colors: Sequence[ParsableManimColor]) -> Paint:
        """The same opacities with other colours (a list makes a gradient)."""
        rgbs = np.array([ManimColor(c).to_rgb() for c in colors], dtype=float)
        opacities = self._rgbas[:, 3]
        length = max(len(rgbs), len(opacities))
        rgbas = np.empty((length, 4))
        rgbas[:, :3] = _stretched(rgbs, length)
        rgbas[:, 3] = _stretched(opacities, length)
        return Paint._from_rgbas(rgbas, getattr(self, "_direction", None))

    @staticmethod
    def _from_rgbas(rgbas: Any, direction: Any = None) -> Paint:
        """A paint from RGBA rows: one row a :class:`Solid`, more a
        :class:`LinearGradient` along ``direction`` (default ``UL``).
        """
        rgbas = _frozen(rgbas)
        rgbas = rgbas.reshape(-1, 4)
        if len(rgbas) == 1:
            paint: Paint = object.__new__(Solid)
        else:
            paint = object.__new__(LinearGradient)
            object.__setattr__(
                paint,
                "_direction",
                _frozen(UL if direction is None else direction, (3,)),
            )
        object.__setattr__(paint, "_rgbas", rgbas)
        return paint


class Solid(Paint):
    """One colour.

    Parameters
    ----------
    color
        The colour.
    opacity
        Its opacity, from 0 (invisible) to 1.
    """

    __slots__ = ()

    def __init__(self, color: ParsableManimColor, opacity: float = 1.0) -> None:
        rgba = ManimColor(color).to_rgba_with_alpha(float(opacity))
        object.__setattr__(self, "_rgbas", _frozen(rgba, (1, 4)))

    @property
    def color(self) -> ManimColor:
        return ManimColor(self._rgbas[0, :3])

    @property
    def opacity(self) -> float:
        return float(self._rgbas[0, 3])

    def __repr__(self) -> str:
        return f"Solid({self.color.to_hex()!r}, opacity={self.opacity:g})"


class LinearGradient(Paint):
    """Evenly spaced colour stops along ``direction``, from one edge of the
    mobject's own bounding box to the other.

    Parameters
    ----------
    colors
        Two or more colours.
    opacities
        One opacity for all stops, or one per stop.
    direction
        The direction the colours change along.
    """

    __slots__ = ("_direction",)
    _direction: Vector3D

    def __init__(
        self,
        colors: Sequence[ParsableManimColor],
        opacities: float | Sequence[float] = 1.0,
        direction: Vector3DLike = UL,
    ) -> None:
        colors = list(colors)
        if len(colors) < 2:
            raise ValueError("A LinearGradient has two or more colors; use Solid.")
        rgbs = np.array([ManimColor(c).to_rgb() for c in colors], dtype=float)
        alphas = np.broadcast_to(np.asarray(opacities, dtype=float), (len(rgbs),))
        object.__setattr__(
            self, "_rgbas", _frozen(np.column_stack([rgbs, alphas]), (len(rgbs), 4))
        )
        object.__setattr__(self, "_direction", _frozen(direction, (3,)))

    @property
    def direction(self) -> Vector3D:
        """The direction the colours change along (read-only)."""
        return self._direction

    def with_direction(self, direction: Vector3DLike) -> LinearGradient:
        """The same stops along another direction."""
        return Paint._from_rgbas(self._rgbas, direction)  # type: ignore[return-value]

    def __repr__(self) -> str:
        colors = [color.to_hex() for color in self.colors]
        return f"LinearGradient({colors}, direction={self._direction.tolist()})"


def _stretched(array: np.ndarray, length: int) -> np.ndarray:
    """``array`` with its rows repeated to ``length`` rows, keeping their order (as
    ``stretch_array_to_length``).
    """
    current = len(array)
    if current == length:
        return np.array(array)
    indices = (np.arange(length) * current / length).astype(int)
    return np.asarray(array)[indices]


def interpolate_paint(start: Paint, end: Paint, alpha: float) -> Paint:
    """The paint at ``alpha`` between ``start`` and ``end``.

    The paints themselves at 0 and 1, and whenever ``start is end``. Otherwise the
    shorter list of stops is stretched to the longer one's length and the colours
    are interpolated linearly; directions interpolate too, a :class:`Solid` taking
    the other paint's direction.
    """
    if start is end or alpha == 1:
        return end
    if alpha == 0:
        return start
    a, b = start._rgbas, end._rgbas
    length = max(len(a), len(b))
    rgbas = (1 - alpha) * _stretched(a, length) + alpha * _stretched(b, length)
    if length == 1:
        return Paint._from_rgbas(rgbas)
    d0 = getattr(start, "_direction", None)
    d1 = getattr(end, "_direction", None)
    if d0 is None:
        direction = d1
    elif d1 is None:
        direction = d0
    else:
        direction = (1 - alpha) * d0 + alpha * d1
    return Paint._from_rgbas(rgbas, direction)
