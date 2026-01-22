#!/usr/bin/env python
from __future__ import annotations

from queue import PriorityQueue
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from manim.typing import (
        Point2D,
        Point2D_Array,
        Point2DLike,
        Point2DLike_Array,
        Point3DLike_Array,
    )


class Polygon:
    """
    Initializes the Polygon with the given rings.

    Parameters
    ----------
    rings
        A sequence of points, where each sequence represents the rings of the polygon.
        Typically, multiple rings indicate holes in the polygon.
    """

    def __init__(self, rings: Sequence[Point2DLike_Array]) -> None:
        np_rings: list[Point2D_Array] = [np.asarray(ring) for ring in rings]
        # Flatten Array
        csum = np.cumsum([ring.shape[0] for ring in np_rings])
        self.array: Point2D_Array = np.concatenate(np_rings, axis=0)

        # Compute Boundary
        self.start: Point2D_Array = np.delete(self.array, csum - 1, axis=0)
        self.stop: Point2D_Array = np.delete(self.array, csum % csum[-1], axis=0)
        self.diff: Point2D_Array = np.delete(
            np.diff(self.array, axis=0), csum[:-1] - 1, axis=0
        )
        self.norm: Point2D_Array = self.diff / np.einsum(
            "ij,ij->i", self.diff, self.diff
        ).reshape(-1, 1)

        # Compute Centroid
        x, y = self.start[:, 0], self.start[:, 1]
        xr, yr = self.stop[:, 0], self.stop[:, 1]
        self.area: float = 0.5 * (np.dot(x, yr) - np.dot(xr, y))
        if self.area:
            factor = x * yr - xr * y
            cx = np.sum((x + xr) * factor) / (6.0 * self.area)
            cy = np.sum((y + yr) * factor) / (6.0 * self.area)
            self.centroid = np.array([cx, cy])

    def compute_distance(self, point: Point2DLike) -> float:
        """Compute the minimum distance from a point to the polygon."""
        scalars = np.einsum("ij,ij->i", self.norm, point - self.start)
        clips = np.clip(scalars, 0, 1).reshape(-1, 1)
        d: float = np.min(
            np.linalg.norm(self.start + self.diff * clips - point, axis=1)
        )
        return d if self.inside(point) else -d

    def _is_point_on_segment(
        self,
        x_point: float,
        y_point: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        """
        Check if a point is on the segment.

        The segment is defined by (x0, y0) to (x1, y1).
        """
        if min(x0, x1) <= x_point <= max(x0, x1) and min(y0, y1) <= y_point <= max(
            y0, y1
        ):
            dx = x1 - x0
            dy = y1 - y0
            cross = dx * (y_point - y0) - dy * (x_point - x0)
            return bool(np.isclose(cross, 0.0))
        return False

    def _ray_crosses_segment(
        self,
        x_point: float,
        y_point: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> bool:
        """
        Check if a horizontal ray to the right from point (x_point, y_point) crosses the segment.

        The segment is defined by (x0, y0) to (x1, y1).
        """
        if (y0 > y_point) != (y1 > y_point):
            slope = (x1 - x0) / (y1 - y0)
            x_intersect = slope * (y_point - y0) + x0
            return bool(x_point < x_intersect)
        return False

    def inside(self, point: Point2DLike) -> bool:
        """
        Check if a point is inside the polygon.

        Uses ray casting algorithm and checks boundary points consistently.
        """
        point_x, point_y = point
        start_x, start_y = self.start[:, 0], self.start[:, 1]
        stop_x, stop_y = self.stop[:, 0], self.stop[:, 1]
        segment_count = len(start_x)

        for i in range(segment_count):
            if self._is_point_on_segment(
                point_x,
                point_y,
                start_x[i],
                start_y[i],
                stop_x[i],
                stop_y[i],
            ):
                return True

        crossings = 0
        for i in range(segment_count):
            if self._ray_crosses_segment(
                point_x,
                point_y,
                start_x[i],
                start_y[i],
                stop_x[i],
                stop_y[i],
            ):
                crossings += 1

        return crossings % 2 == 1


class Cell:
    """
    A square in a mesh covering the :class:`~.Polygon` passed as an argument.

    Parameters
    ----------
    c
        Center coordinates of the Cell.
    h
        Half-Size of the Cell.
    polygon
        :class:`~.Polygon` object for which the distance is computed.
    """

    def __init__(self, c: Point2DLike, h: float, polygon: Polygon) -> None:
        self.c: Point2D = np.asarray(c)
        self.h = h
        self.d = polygon.compute_distance(self.c)
        self.p = self.d + self.h * np.sqrt(2)

    def __lt__(self, other: Cell) -> bool:
        return self.d < other.d

    def __gt__(self, other: Cell) -> bool:
        return self.d > other.d

    def __le__(self, other: Cell) -> bool:
        return self.d <= other.d

    def __ge__(self, other: Cell) -> bool:
        return self.d >= other.d


def polylabel(rings: Sequence[Point3DLike_Array], precision: float = 0.01) -> Cell:
    """
    Finds the pole of inaccessibility (the point that is farthest from the edges of the polygon)
    using an iterative grid-based approach.

    Parameters
    ----------
    rings
        A list of lists, where each list is a sequence of points representing the rings of the polygon.
        Typically, multiple rings indicate holes in the polygon.
    precision
        The precision of the result (default is 0.01).

    Returns
    -------
    Cell
        A Cell containing the pole of inaccessibility to a given precision.
    """
    # Precompute Polygon Data
    np_rings: list[Point2D_Array] = [np.asarray(ring)[:, :2] for ring in rings]
    polygon = Polygon(np_rings)

    # Bounding Box
    mins = np.min(polygon.array, axis=0)
    maxs = np.max(polygon.array, axis=0)
    dims = maxs - mins
    s = np.min(dims)
    h = s / 2.0

    # Initial Grid
    queue: PriorityQueue[Cell] = PriorityQueue()
    xv, yv = np.meshgrid(np.arange(mins[0], maxs[0], s), np.arange(mins[1], maxs[1], s))
    for corner in np.vstack([xv.ravel(), yv.ravel()]).T:
        queue.put(Cell(corner + h, h, polygon))

    # Initial Guess
    best = Cell(polygon.centroid, 0, polygon)
    bbox = Cell(mins + (dims / 2), 0, polygon)
    if bbox.d > best.d:
        best = bbox

    # While there are cells to consider...
    directions = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]])
    while not queue.empty():
        cell = queue.get()
        if cell > best:
            best = cell
        # If a cell is promising, subdivide!
        if cell.p - best.d > precision:
            h = cell.h / 2.0
            offsets = cell.c + directions * h
            queue.put(Cell(offsets[0], h, polygon))
            queue.put(Cell(offsets[1], h, polygon))
            queue.put(Cell(offsets[2], h, polygon))
            queue.put(Cell(offsets[3], h, polygon))
    return best
