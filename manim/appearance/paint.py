"""Paints: how an area or a line of a vectorized mobject is colored.

A paint is an immutable value. :class:`SolidColor` is one color;
:class:`LinearGradient` is evenly spaced color stops along a direction, between the
edges of the mobject's own bounding box. Assign one to a mobject's ``fill``,
``stroke`` or ``background_stroke``, or let :meth:`.VMobject.set_fill` and
:meth:`.VMobject.set_stroke` make it::

    square.fill = LinearGradient([RED, BLUE], direction=RIGHT)
    square.set_fill([RED, BLUE], opacity=1)  # the same, with the default direction

Paints of your own subclass :class:`Paint`: :meth:`Paint.resolve` gives what
renderers draw (a render snapshot paint of :mod:`manim.renderer.snapshot`), and
:meth:`Paint.interpolate` animates it. A paint that resolves to the built-in
snapshot paints is drawn by every renderer; renderers raise
:class:`~.CapabilityError` for snapshot paints they do not know.

Paints are compared by identity: two equal paints made separately are two values.
A gradient's end points are recomputed from the mobject's current points (its own
bounding box projected on ``direction``); :meth:`.VMobject.rotate` turns the
direction, but on a shape that is not square the end points do not turn rigidly
with it.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from manim.constants import UL
from manim.utils.color import ManimColor, ParsableManimColor

if TYPE_CHECKING:
    from typing import Self

    from manim.renderer.snapshot import RPaint
    from manim.typing import FloatRGBA_Array, Point3D_Array, Vector3D, Vector3DLike

__all__ = ["ColorField", "LinearGradient", "Paint", "SolidColor", "interpolate_paint"]


def _frozen(array: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.array(array, dtype=np.float64)
    if shape is not None:
        result = result.reshape(shape)
    result.flags.writeable = False
    return result


class Paint:
    """A coloring of an area or a line; immutable.

    See :mod:`manim.appearance.paint`.
    """

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
        """The color stops as a read-only float array of RGBA rows (one row for a
        :class:`SolidColor`).
        """
        return self._rgbas

    @property
    def colors(self) -> list[ManimColor]:
        """The colors of the stops."""
        return [ManimColor(rgba[:3]) for rgba in self._rgbas]

    @property
    def opacities(self) -> np.ndarray:
        """The opacities of the stops (read-only)."""
        return self._rgbas[:, 3]

    def with_opacity(self, opacity: float | Sequence[float]) -> Paint:
        """The same colors with ``opacity`` (one for all stops, or one per stop)."""
        rgbas = self._rgbas.copy()
        opacities = np.asarray(opacity, dtype=float)
        if opacities.ndim and len(opacities) != len(rgbas):
            rgbas = _stretched(rgbas, max(len(rgbas), len(opacities)))
            opacities = _stretched(opacities, len(rgbas))
        rgbas[:, 3] = opacities
        return Paint._from_rgbas(rgbas, getattr(self, "_direction", None))

    def with_colors(self, colors: Sequence[ParsableManimColor]) -> Paint:
        """The same opacities with other colors (a list makes a gradient)."""
        rgbs = np.array([ManimColor(c).to_rgb() for c in colors], dtype=float)
        opacities = self._rgbas[:, 3]
        length = max(len(rgbs), len(opacities))
        rgbas = np.empty((length, 4))
        rgbas[:, :3] = _stretched(rgbs, length)
        rgbas[:, 3] = _stretched(opacities, length)
        return Paint._from_rgbas(rgbas, getattr(self, "_direction", None))

    def resolve(self, points: Point3D_Array) -> Any:
        """What renderers draw for this paint on a path with ``points`` (its own,
        in the scene): a snapshot paint (:data:`~.RPaint`). Called when the path's
        snapshot is made.

        Subclasses override it; the built-in paints resolve to
        :class:`~.RSolidColor`, :class:`~.RLinearGradient` and
        :class:`~.RColorField`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not say what renderers draw: override "
            "Paint.resolve."
        )

    def interpolate(self, end: Paint, alpha: float) -> Paint:
        """The paint at progress ``alpha`` (0 to 1) from this one to ``end``.

        By default it switches to ``end`` when ``alpha`` reaches 1; subclasses
        override it to blend. Called by :func:`interpolate_paint` for ``alpha``
        strictly between 0 and 1 and paints that are not the same object.
        """
        return end if alpha >= 1 else self

    @staticmethod
    def _from_rgbas(rgbas: Any, direction: Any = None) -> Paint:
        """A paint from RGBA rows: one row a :class:`SolidColor`, more a
        :class:`LinearGradient` along ``direction`` (default ``UL``).
        """
        rgbas = _frozen(rgbas)
        rgbas = rgbas.reshape(-1, 4)
        if len(rgbas) == 1:
            paint: Paint = object.__new__(SolidColor)
        else:
            paint = object.__new__(LinearGradient)
            object.__setattr__(
                paint,
                "_direction",
                _frozen(UL if direction is None else direction, (3,)),
            )
        object.__setattr__(paint, "_rgbas", rgbas)
        return paint


class SolidColor(Paint):
    """One color.

    Parameters
    ----------
    color
        The color.
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

    def resolve(self, points: Point3D_Array) -> RPaint:
        from manim.renderer.snapshot import RSolidColor

        return RSolidColor(self._rgbas[0])

    def interpolate(self, end: Paint, alpha: float) -> Paint:
        return _blend_stops(self, end, alpha)

    def __repr__(self) -> str:
        return f"SolidColor({self.color.to_hex()!r}, opacity={self.opacity:g})"


class LinearGradient(Paint):
    """Evenly spaced color stops along ``direction``, from one edge of the
    mobject's own bounding box to the other.

    Parameters
    ----------
    colors
        Two or more colors.
    opacities
        One opacity for all stops, or one per stop.
    direction
        The direction the colors change along.
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
            raise ValueError("A LinearGradient has two or more colors; use SolidColor.")
        rgbs = np.array([ManimColor(c).to_rgb() for c in colors], dtype=float)
        alphas = np.broadcast_to(np.asarray(opacities, dtype=float), (len(rgbs),))
        object.__setattr__(
            self, "_rgbas", _frozen(np.column_stack([rgbs, alphas]), (len(rgbs), 4))
        )
        object.__setattr__(self, "_direction", _frozen(direction, (3,)))

    @property
    def direction(self) -> Vector3D:
        """The direction the colors change along (read-only)."""
        return self._direction

    def with_direction(self, direction: Vector3DLike) -> LinearGradient:
        """The same stops along another direction."""
        return Paint._from_rgbas(self._rgbas, direction)  # type: ignore[return-value]

    def resolve(self, points: Point3D_Array) -> RPaint:
        """The stops spaced between the edges of the bounding box of the path's
        anchors along :attr:`direction`.
        """
        from manim.renderer.snapshot import RLinearGradient

        start, end = _gradient_ends(self._direction, points)
        return RLinearGradient(start, end, self._rgbas)

    def interpolate(self, end: Paint, alpha: float) -> Paint:
        return _blend_stops(self, end, alpha)

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


def _gradient_ends(
    direction: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Where a path's gradient starts and ends: the edges of the bounding box of
    its anchors (the first and last point of every curve) along ``direction``.
    """
    if len(points) == 0:
        return np.zeros(3), np.zeros(3)
    anchors = np.concatenate([points[::4], points[3::4]]) if len(points) > 1 else points
    low, high = anchors.min(axis=0), anchors.max(axis=0)
    center = (low + high) / 2
    offset = (high - low) / 2 * np.asarray(direction, dtype=float)
    return center - offset, center + offset


def interpolate_paint(start: Paint, end: Paint, alpha: float) -> Paint:
    """The paint at ``alpha`` between ``start`` and ``end``: the paints themselves
    at 0 and 1, and whenever ``start is end``; otherwise
    ``start.interpolate(end, alpha)`` (:meth:`Paint.interpolate`).

    Between two built-in color paints (:class:`SolidColor`, :class:`LinearGradient`)
    the shorter list of stops is stretched to the longer one's length and the
    colors are interpolated linearly; directions interpolate too, a
    :class:`SolidColor` taking the other paint's direction. Two :class:`ColorField`
    paints with the same grid and corners interpolate their opacity. Any other pair
    keeps ``start`` until ``alpha`` reaches 1, where it is ``end``.
    """
    if start is end or alpha >= 1:
        return end
    if alpha <= 0:
        return start
    return start.interpolate(end, alpha)


def _blend_stops(start: Paint, end: Paint, alpha: float) -> Paint:
    """Between two color-stop paints (:class:`SolidColor`, :class:`LinearGradient`),
    stops and directions blended (see :func:`interpolate_paint`); towards any other
    paint, ``start`` until the end.
    """
    if not isinstance(end, (SolidColor, LinearGradient)):
        return Paint.interpolate(start, end, alpha)
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


_grid_ids = itertools.count(1)


class ColorField(Paint):
    """Colors that lie in the scene, not on the mobject: a grid of RGBA colors
    and where it is. A mobject painted with a field shows the field's color
    wherever it is drawn, so moving, trimming or transforming it shows other
    colors (as stream lines flowing through a colored vector field).

    Parameters
    ----------
    grid
        RGBA colors, shape ``(rows, columns, 4)``, floats in ``[0, 1]``,
        straight (not premultiplied) alpha, sRGB; row 0 is the top.
    corners
        Three scene points: the upper left, upper right and lower left corner of
        the grid's outer edges. Texel ``(i, j)`` is centred at
        ``((j + 0.5) / columns, (i + 0.5) / rows)`` between them.
    opacity
        Multiplies the grid's alpha.

    Notes
    -----
    The color at a scene point: its position in the plane of the corners
    (projected orthogonally onto it), clamped to the outermost texel centres,
    bilinear interpolation of premultiplied colors, then unpremultiplied and the
    alpha times ``opacity`` (:meth:`sample`). Renderers sample where they draw.
    The Cairo and OpenGL renderers draw fields in the frame's plane under a flat
    camera, and raise :class:`~.CapabilityError` otherwise, and for mobjects fixed
    in the frame or with a fixed orientation.
    """

    __slots__ = ("_corners", "_grid", "_grid_id", "_opacity")
    _grid: np.ndarray
    _corners: np.ndarray
    _opacity: float
    _grid_id: int

    def __init__(self, grid: Any, corners: Any, opacity: float = 1.0) -> None:
        grid = _frozen(grid)
        if grid.ndim != 3 or grid.shape[2] != 4 or min(grid.shape[:2]) < 1:
            raise ValueError("A ColorField's grid has shape (rows, columns, 4).")
        corners = _frozen(corners, (3, 3))
        object.__setattr__(self, "_grid", grid)
        object.__setattr__(self, "_corners", corners)
        object.__setattr__(self, "_opacity", float(opacity))
        object.__setattr__(self, "_grid_id", next(_grid_ids))
        # One stop: the grid's mean color, for getters that ask for colors.
        mean = grid.reshape(-1, 4).mean(axis=0)
        object.__setattr__(
            self, "_rgbas", _frozen(np.r_[mean[:3], mean[3] * opacity], (1, 4))
        )

    @classmethod
    def from_function(
        cls,
        func: Callable[[Any], Any],
        x_range: Sequence[float],
        y_range: Sequence[float],
        resolution: tuple[int, int],
        *,
        vectorized: bool = False,
        z: float = 0.0,
        opacity: float = 1.0,
    ) -> ColorField:
        """A field in the plane ``z`` over ``x_range`` x ``y_range``, sampled at
        the centres of ``resolution`` = (columns, rows) texels.

        ``func`` takes one point and returns one RGB or RGBA color (floats in
        ``[0, 1]``, or anything :class:`.ManimColor` takes); with ``vectorized``,
        it takes an ``(n, 3)`` array and returns ``(n, 3)`` or ``(n, 4)`` floats.
        """
        columns, rows = (int(n) for n in resolution)
        if columns < 1 or rows < 1:
            raise ValueError("A ColorField has at least one texel.")
        x0, x1 = float(x_range[0]), float(x_range[1])
        y0, y1 = float(y_range[0]), float(y_range[1])
        xs = x0 + (np.arange(columns) + 0.5) * (x1 - x0) / columns
        ys = y1 - (np.arange(rows) + 0.5) * (y1 - y0) / rows
        gx, gy = np.meshgrid(xs, ys)
        points = np.stack([gx, gy, np.full_like(gx, z)], axis=-1).reshape(-1, 3)
        if vectorized:
            colors = np.asarray(func(points), dtype=float)
        else:
            colors = np.array([_rgba_of(func(point)) for point in points])
        if colors.shape[1] == 3:
            colors = np.column_stack([colors, np.ones(len(colors))])
        return cls(
            colors.reshape(rows, columns, 4),
            [[x0, y1, z], [x1, y1, z], [x0, y0, z]],
            opacity,
        )

    @property
    def grid(self) -> np.ndarray:
        """The colors (read-only), shape ``(rows, columns, 4)``."""
        return self._grid

    @property
    def corners(self) -> np.ndarray:
        """Upper left, upper right and lower left corner (read-only)."""
        return self._corners

    @property
    def opacity(self) -> float:
        return self._opacity

    @property
    def grid_id(self) -> int:
        """Identifies the grid's storage: equal ids, the same colors."""
        return self._grid_id

    def with_opacity(self, opacity: float | Sequence[float]) -> ColorField:
        """The same grid and placement with another ``opacity``."""
        value = float(np.asarray(opacity, dtype=float).reshape(-1)[0])
        field = object.__new__(ColorField)
        for name in ("_grid", "_corners", "_grid_id"):
            object.__setattr__(field, name, getattr(self, name))
        object.__setattr__(field, "_opacity", value)
        mean = self._rgbas[0]
        alpha = self._grid[..., 3].mean() * value
        object.__setattr__(field, "_rgbas", _frozen(np.r_[mean[:3], alpha], (1, 4)))
        return field

    def with_colors(self, colors: Sequence[ParsableManimColor]) -> Paint:
        """Colors replace the field: a solid or a gradient with its opacity."""
        rgbs = np.array([ManimColor(c).to_rgb() for c in colors], dtype=float)
        rgbas = np.column_stack([rgbs, np.full(len(rgbs), self._rgbas[0, 3])])
        return Paint._from_rgbas(rgbas)

    def same_grid(self, other: object) -> bool:
        """Whether ``other`` is a field with this grid and placement."""
        return (
            isinstance(other, ColorField)
            and other._grid_id == self._grid_id
            and np.array_equal(other._corners, self._corners)
        )

    def sample(self, points: Any) -> np.ndarray:
        """RGBA (straight alpha) at scene points: the reference every renderer
        follows.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        rows, columns = self._grid.shape[:2]
        upper_left, upper_right, lower_left = self._corners
        eu, ev = upper_right - upper_left, lower_left - upper_left
        offset = points - upper_left
        u = offset @ eu / (eu @ eu)
        v = offset @ ev / (ev @ ev)
        x = np.clip(u * columns - 0.5, 0, columns - 1)
        y = np.clip(v * rows - 0.5, 0, rows - 1)
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        x1, y1 = np.minimum(x0 + 1, columns - 1), np.minimum(y0 + 1, rows - 1)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        premultiplied = np.array(self._grid)
        premultiplied[..., :3] *= premultiplied[..., 3:4]
        top = premultiplied[y0, x0] * (1 - fx) + premultiplied[y0, x1] * fx
        bottom = premultiplied[y1, x0] * (1 - fx) + premultiplied[y1, x1] * fx
        out = top * (1 - fy) + bottom * fy
        alpha = out[:, 3:4]
        rgb = np.where(alpha > 0, out[:, :3] / np.where(alpha > 0, alpha, 1), 0.0)
        return np.concatenate([rgb, alpha * self._opacity], axis=1)

    def resolve(self, points: Point3D_Array) -> RPaint:
        from manim.renderer.snapshot import RColorField

        return RColorField(self._grid, self._corners, self._opacity, self._grid_id)

    def interpolate(self, end: Paint, alpha: float) -> Paint:
        """Towards a field on the same grid, the opacity fades; towards any other
        paint, the field stays until the end.
        """
        if isinstance(end, ColorField) and self.same_grid(end):
            return self.with_opacity((1 - alpha) * self._opacity + alpha * end.opacity)
        return Paint.interpolate(self, end, alpha)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            ColorField,
            (np.array(self._grid), np.array(self._corners), self._opacity),
        )

    def __repr__(self) -> str:
        rows, columns = self._grid.shape[:2]
        return f"ColorField({rows}x{columns}, opacity={self._opacity:g})"


def _rgba_of(color: Any) -> np.ndarray:
    if isinstance(color, (str, ManimColor)):
        return np.asarray(ManimColor(color).to_rgba(), dtype=float)
    array = np.asarray(color, dtype=float).reshape(-1)
    return array if len(array) == 4 else np.r_[array[:3], 1.0]
