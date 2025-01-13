#!/usr/bin/env python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from manim.typing import PointND, PointND_Array


class QuickHullPoint:
    def __init__(self, coordinates: PointND_Array) -> None:
        self.coordinates = coordinates

    def __hash__(self) -> int:
        return hash(self.coordinates.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuickHullPoint):
            raise ValueError
        are_coordinates_equal: bool = np.array_equal(
            self.coordinates, other.coordinates
        )
        return are_coordinates_equal


class SubFacet:
    def __init__(self, coordinates: PointND_Array) -> None:
        self.coordinates = coordinates
        self.points = frozenset(QuickHullPoint(c) for c in coordinates)

    def __hash__(self) -> int:
        return hash(self.points)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SubFacet):
            raise ValueError
        return self.points == other.points


class Facet:
    def __init__(self, coordinates: PointND_Array, internal: PointND) -> None:
        self.coordinates = coordinates
        self.center: PointND = np.mean(coordinates, axis=0)
        self.normal = self.compute_normal(internal)
        self.subfacets = frozenset(
            SubFacet(np.delete(self.coordinates, i, axis=0))
            for i in range(self.coordinates.shape[0])
        )

    def compute_normal(self, internal: PointND) -> PointND:
        centered = self.coordinates - self.center
        _, _, vh = np.linalg.svd(centered)
        normal: PointND = vh[-1, :]
        normal /= np.linalg.norm(normal)

        # If the normal points towards the internal point, flip it!
        if np.dot(normal, self.center - internal) < 0:
            normal *= -1

        return normal

    def __hash__(self) -> int:
        return hash(self.subfacets)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Facet):
            raise ValueError
        return self.subfacets == other.subfacets


class Horizon:
    def __init__(self) -> None:
        self.facets: set[Facet] = set()
        self.boundary: list[SubFacet] = []


class QuickHull:
    """
    QuickHull algorithm for constructing a convex hull from a set of points.

    Parameters
    ----------
    tolerance
        A tolerance threshold for determining when points lie on the convex hull (default is 1e-5).

    Attributes
    ----------
    facets
        List of facets considered.
    removed
        Set of internal facets that have been removed from the hull during the construction process.
    outside
        Dictionary mapping each facet to its outside points and eye point.
    neighbors
        Mapping of subfacets to their neighboring facets. Each subfacet links precisely two neighbors.
    unclaimed
        Points that have not yet been classified as inside or outside the current hull.
    internal
        An internal point (i.e., the center of the initial simplex) used as a reference during hull construction.
    tolerance
        The tolerance used to determine if points are considered outside the current hull.
    """

    def __init__(self, tolerance: float = 1e-5) -> None:
        self.facets: list[Facet] = []
        self.removed: set[Facet] = set()
        self.outside: dict[Facet, tuple[PointND_Array | None, PointND | None]] = {}
        self.neighbors: dict[SubFacet, set[Facet]] = {}
        self.unclaimed: PointND_Array | None = None
        self.internal: PointND | None = None
        self.tolerance = tolerance

    def initialize(self, points: PointND_Array) -> None:
        # Sample Points
        simplex = points[
            np.random.choice(points.shape[0], points.shape[1] + 1, replace=False)
        ]
        self.unclaimed = points
        new_internal: PointND = np.mean(simplex, axis=0)
        self.internal = new_internal

        # Build Simplex
        for c in range(simplex.shape[0]):
            facet = Facet(np.delete(simplex, c, axis=0), internal=new_internal)
            self.classify(facet)
            self.facets.append(facet)

        # Attach Neighbors
        for f in self.facets:
            for sf in f.subfacets:
                self.neighbors.setdefault(sf, set()).add(f)

    def classify(self, facet: Facet) -> None:
        assert self.unclaimed is not None, (
            "Call .initialize() before using .classify()."
        )

        if not self.unclaimed.size:
            self.outside[facet] = (None, None)
            return

        # Compute Projections
        projections = (self.unclaimed - facet.center) @ facet.normal
        arg = np.argmax(projections)
        mask = projections > self.tolerance

        # Identify Eye and Outside Set
        eye = self.unclaimed[arg] if projections[arg] > self.tolerance else None
        outside = self.unclaimed[mask]
        self.outside[facet] = (outside, eye)
        self.unclaimed = self.unclaimed[~mask]

    def compute_horizon(self, eye: PointND, start_facet: Facet) -> Horizon:
        horizon = Horizon()
        self._recursive_horizon(eye, start_facet, horizon)
        return horizon

    def _recursive_horizon(self, eye: PointND, facet: Facet, horizon: Horizon) -> bool:
        visible = np.dot(facet.normal, eye - facet.center) > 0
        if not visible:
            return False

        # If the eye is visible from the facet:
        # Label the facet as visible and cross each edge
        horizon.facets.add(facet)
        for subfacet in facet.subfacets:
            neighbor = (self.neighbors[subfacet] - {facet}).pop()
            # If the neighbor is not visible, then the edge shared must be on the boundary
            if neighbor not in horizon.facets and not self._recursive_horizon(
                eye, neighbor, horizon
            ):
                horizon.boundary.append(subfacet)
        return True

    def build(self, points: PointND_Array) -> None:
        num, dim = points.shape
        if (dim == 0) or (num < dim + 1):
            raise ValueError("Not enough points supplied to build Convex Hull!")
        if dim == 1:
            raise ValueError("The Convex Hull of 1D data is its min-max!")

        self.initialize(points)

        # This helps the type checker.
        assert self.unclaimed is not None
        assert self.internal is not None

        while True:
            updated = False
            for facet in self.facets:
                if facet in self.removed:
                    continue
                outside, eye = self.outside[facet]
                if eye is not None:
                    updated = True
                    horizon = self.compute_horizon(eye, facet)
                    for f in horizon.facets:
                        points_to_append = self.outside[f][0]
                        # TODO: is this always true?
                        assert points_to_append is not None
                        self.unclaimed = np.vstack((self.unclaimed, points_to_append))
                        self.removed.add(f)
                        for sf in f.subfacets:
                            self.neighbors[sf].discard(f)
                            if self.neighbors[sf] == set():
                                del self.neighbors[sf]
                    for sf in horizon.boundary:
                        nf = Facet(
                            np.vstack((sf.coordinates, eye)), internal=self.internal
                        )
                        self.classify(nf)
                        self.facets.append(nf)
                        for nsf in nf.subfacets:
                            self.neighbors.setdefault(nsf, set()).add(nf)
            if not updated:
                break
