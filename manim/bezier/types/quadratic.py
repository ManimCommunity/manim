from __future__ import annotations

__all__ = ["ManimQuadraticBezier"]

import sys
import typing
from functools import reduce
from typing import Iterable

import numpy as np

from ...utils.simple_functions import choose
from ...utils.space_ops import cross2d, find_intersection
from ..bezier import ManimBezier


class ManimQuadraticBezier(ManimBezier):
    degree = 2

    @staticmethod
    def partial_bezier_points(points, a, b):
        """Shortened version of partial_bezier_points just for quadratics,
        since this is called a fair amount.

        To see an explanation, see split_bezier.

        Parameters
        ----------
        points
            set of points defining the quadratic Bézier curve.
        a
            lower bound of the desired partial quadratic Bézier curve.
        b
            upper bound of the desired partial quadratic Bézier curve.

        Returns
        -------
        np.ndarray
            Set of points defining the partial quadratic Bézier curve.
        """

        points = np.asarray(points, dtype=np.float64)
        if a == 1:
            return 3 * [points[-1]]

        def curve(t):
            return (
                points[0] * (1 - t) * (1 - t)
                + 2 * points[1] * t * (1 - t)
                + points[2] * t * t
            )

        # bezier(points)
        h0 = curve(a) if a > 0 else points[0]
        h2 = curve(b) if b < 1 else points[2]
        h1_prime = (1 - a) * points[1] + a * points[2]
        end_prop = (b - a) / (1.0 - a)
        h1 = (1 - end_prop) * h0 + end_prop * h1_prime
        return [h0, h1, h2]

    @classmethod
    def split_bezier(cls, points: np.ndarray, t: float) -> np.ndarray:
        """Split a quadratic Bézier curve at argument ``t`` into two quadratic curves.

        Parameters
        ----------
        points
            The control points of the bezier curve
            has shape ``[a1, h1, b1]``

        t
            The ``t``-value at which to split the Bézier curve

        Returns
        -------
            The two Bézier curves as a list of tuples,
            has the shape ``[a1, h1, b1], [a2, h2, b2]``
        """
        a1, h1, a2 = points
        s1 = cls.interpolate(a1, h1, t)
        s2 = cls.interpolate(h1, a2, t)
        p = cls.interpolate(s1, s2, t)

        return np.array([a1, s1, p, p, s2, a2])

    @classmethod
    def subdivide_bezier(cls, points: Iterable[float], n: int) -> np.ndarray:
        """Subdivide a quadratic Bézier curve into ``n`` subcurves which have the same shape.

        The points at which the curve is split are located at the
        arguments :math:`t = i/n` for :math:`i = 1, ..., n-1`.

        Parameters
        ----------
        points
            The control points of the Bézier curve in form ``[a1, h1, b1]``

        n
            The number of curves to subdivide the Bézier curve into

        Returns
        -------
            The new points for the Bézier curve in the form ``[a1, h1, b1, a2, h2, b2, ...]``

        .. image:: /_static/bezier_subdivision_example.png

        """
        beziers = np.empty((n, 3, 3))
        current = points
        for j in range(0, n):
            i = n - j
            tmp = cls.split_bezier(current, 1 / i)
            beziers[j] = tmp[:3]
            current = tmp[3:]
        return beziers.reshape(-1, 3)

    @classmethod
    def bezier_remap(
        cls,
        triplets: np.ndarray,
        new_number_of_curves: int,
    ):
        """Remaps the number of curves to a higher amount by splitting bezier curves

        Parameters
        ----------
        triplets
            The triplets of the quadratic bezier curves to be remapped shape(n, 3, 3)

        new_number_of_curves
            The number of curves that the output will contain. This needs to be higher than the current number.

        Returns
        -------
            The new triplets for the quadratic bezier curves.
        """
        difference = new_number_of_curves - len(triplets)
        if difference <= 0:
            return triplets
        new_triplets = np.zeros((new_number_of_curves, 3, 3))
        idx = 0
        for triplet in triplets:
            if difference > 0:
                tmp_noc = int(np.ceil(difference / len(triplets))) + 1
                tmp = cls.subdivide_bezier(triplet, tmp_noc).reshape(-1, 3, 3)
                for i in range(tmp_noc):
                    new_triplets[idx + i] = tmp[i]
                difference -= tmp_noc - 1
                idx += tmp_noc
            else:
                new_triplets[idx] = triplet
                idx += 1
        return new_triplets

        """
        This is an alternate version of the function just for documentation purposes
        --------

        difference = new_number_of_curves - len(triplets)
        if difference <= 0:
            return triplets
        new_triplets = []
        for triplet in triplets:
            if difference > 0:
                tmp_noc = int(np.ceil(difference / len(triplets))) + 1
                tmp = subdivide_quadratic_bezier(triplet, tmp_noc).reshape(-1, 3, 3)
                for i in range(tmp_noc):
                    new_triplets.append(tmp[i])
                difference -= tmp_noc - 1
            else:
                new_triplets.append(triplet)
        return new_triplets
        """
