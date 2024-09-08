#!/usr/bin/env python
from __future__ import annotations

import numpy as np


class Point:
    def __init__(self, coordinates):
        self.coordinates = coordinates

    def __hash__(self):
        return hash(self.coordinates.tobytes())

    def __eq__(self, other):
        return np.array_equal(self.coordinates, other.coordinates)


class SubFacet:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.points = frozenset(Point(c) for c in coordinates)

    def __hash__(self):
        return hash(self.points)

    def __eq__(self, other):
        return self.points == other.points


class Facet:
    def __init__(self, coordinates, normal=None, internal=None):
        self.coordinates = coordinates
        self.center = np.mean(coordinates, axis=0)
        self.normal = self.compute_normal(internal)
        self.subfacets = frozenset(
            SubFacet(np.delete(self.coordinates, i, axis=0))
            for i in range(self.coordinates.shape[0])
        )

    def compute_normal(self, internal):
        centered = self.coordinates - self.center
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1, :]
        normal /= np.linalg.norm(normal)

        if np.dot(normal, self.center - internal) < 0:
            normal *= -1

        return normal

    def __hash__(self):
        return hash(self.subfacets)

    def __eq__(self, other):
        return self.subfacets == other.subfacets


class Horizon:
    def __init__(self):
        self.facets = set()
        self.boundary = []


class QuickHull:
    def __init__(self, tolerance=1e-5):
        self.facets = []
        self.removed = set()
        self.outside = {}
        self.neighbors = {}
        self.unclaimed = None
        self.internal = None
        self.tolerance = tolerance

    def initialize(self, points):
        # Sample Points
        simplex = points[
            np.random.choice(points.shape[0], points.shape[1] + 1, replace=False)
        ]
        self.unclaimed = points
        self.internal = np.mean(simplex, axis=0)

        # Build Simplex
        for c in range(simplex.shape[0]):
            facet = Facet(np.delete(simplex, c, axis=0), internal=self.internal)
            self.classify(facet)
            self.facets.append(facet)

        # Attach Neighbors
        for f in self.facets:
            for sf in f.subfacets:
                self.neighbors.setdefault(sf, set()).add(f)

    def classify(self, facet):
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

    def compute_horizon(self, eye, start_facet):
        horizon = Horizon()
        self._recursive_horizon(eye, start_facet, horizon)
        return horizon

    def _recursive_horizon(self, eye, facet, horizon):
        # If the eye is visible from the facet...
        if np.dot(facet.normal, eye - facet.center) > 0:
            # Label the facet as visible and cross each edge
            horizon.facets.add(facet)
            for subfacet in facet.subfacets:
                neighbor = (self.neighbors[subfacet] - {facet}).pop()
                # If the neighbor is not visible, then the edge shared must be on the boundary
                if neighbor not in horizon.facets and not self._recursive_horizon(
                    eye, neighbor, horizon
                ):
                    horizon.boundary.append(subfacet)
            return 1
        else:
            return 0

    def build(self, points):
        num, dim = points.shape
        if (dim == 0) or (num < dim + 1):
            raise ValueError("Not enough points supplied to build Convex Hull!")
        if dim == 1:
            return np.array([np.min(points), np.max(points)])

        self.initialize(points)
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
                        self.unclaimed = np.vstack((self.unclaimed, self.outside[f][0]))
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
