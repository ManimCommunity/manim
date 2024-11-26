#!/usr/bin/env python
from __future__ import annotations

from queue import PriorityQueue
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manim.typing import Point3D, Point3D_Array


class Polygon:
    """
    Initializes the Polygon with the given rings.

    Parameters
    ----------
    rings
        A collection of closed polygonal ring.
    """

    def __init__(self, rings: Point3D_Array) -> None:
        # Flatten Array
        csum = np.cumsum([ring.shape[0] for ring in rings])
        self.array = np.concatenate(rings, axis=0)

        # Compute Boundary
        self.start = np.delete(self.array, csum - 1, axis=0)
        self.stop = np.delete(self.array, csum % csum[-1], axis=0)
        self.diff = np.delete(np.diff(self.array, axis=0), csum[:-1] - 1, axis=0)
        self.norm = self.diff / np.einsum("ij,ij->i", self.diff, self.diff).reshape(
            -1, 1
        )

        # Compute Centroid
        x, y = self.start[:, 0], self.start[:, 1]
        xr, yr = self.stop[:, 0], self.stop[:, 1]
        self.area = 0.5 * (np.dot(x, yr) - np.dot(xr, y))
        if self.area:
            factor = x * yr - xr * y
            cx = np.sum((x + xr) * factor) / (6.0 * self.area)
            cy = np.sum((y + yr) * factor) / (6.0 * self.area)
            self.centroid = np.array([cx, cy])

    def compute_distance(self, point: Point3D) -> float:
        """Compute the minimum distance from a point to the polygon."""
        scalars = np.einsum("ij,ij->i", self.norm, point - self.start)
        clips = np.clip(scalars, 0, 1).reshape(-1, 1)
        d = np.min(np.linalg.norm(self.start + self.diff * clips - point, axis=1))
        return d if self.inside(point) else -d

    def inside(self, point: Point3D) -> bool:
        """Check if a point is inside the polygon."""
        # Views
        px, py = point
        x, y = self.start[:, 0], self.start[:, 1]
        xr, yr = self.stop[:, 0], self.stop[:, 1]

        # Count Crossings (enforce short-circuit)
        c = (y > py) != (yr > py)
        c = px < x[c] + (py - y[c]) * (xr[c] - x[c]) / (yr[c] - y[c])
        return np.sum(c) % 2 == 1


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

    def __init__(self, c: Point3D, h: float, polygon: Polygon) -> None:
        self.c = c
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


def polylabel(rings: Point3D_Array, precision: float = 0.01) -> Cell:
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
    array = [np.array(ring)[:, :2] for ring in rings]
    polygon = Polygon(array)

    # Bounding Box
    mins = np.min(polygon.array, axis=0)
    maxs = np.max(polygon.array, axis=0)
    dims = maxs - mins
    s = np.min(dims)
    h = s / 2.0

    # Initial Grid
    queue = PriorityQueue()
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
