from __future__ import annotations

__all__ = ["ManimQuadraticBezier"]

import numpy as np

from ...utils.simple_functions import choose
from ...utils.space_ops import cross2d, find_intersection
from ..bezier import ManimBezier


class ManimQuadraticBezier(ManimBezier):
    degree = 2

    subdivision_matrices = {
        2: np.array(
            [
                [4, 0, 0],
                [2, 2, 0],
                [1, 2, 1],
                [1, 2, 1],
                [0, 2, 2],
                [0, 0, 4],
            ]
        )
        / 4,
        3: np.array(
            [
                [9, 0, 0],
                [6, 3, 0],
                [4, 4, 1],
                [4, 4, 1],
                [2, 5, 2],
                [1, 4, 4],
                [1, 4, 4],
                [0, 3, 6],
                [0, 0, 9],
            ]
        )
        / 9,
        4: np.array(
            [
                [16, 0, 0],
                [12, 4, 0],
                [9, 6, 1],
                [9, 6, 1],
                [6, 8, 2],
                [4, 8, 4],
                [4, 8, 4],
                [2, 8, 6],
                [1, 6, 9],
                [1, 6, 9],
                [0, 4, 12],
                [0, 0, 16],
            ]
        )
        / 16,
    }

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

    @staticmethod
    def split_bezier(points: np.ndarray, t: float) -> np.ndarray:
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
        mt = 1 - t
        mt2 = mt * mt
        t2 = t * t
        two_mt_t = 2 * mt * t

        split_matrix = np.array(
            [
                [1, 0, 0],
                [mt, t, 0],
                [mt2, two_mt_t, t2],
                [mt2, two_mt_t, t2],
                [0, mt, t],
                [0, 0, 1],
            ]
        )

        return split_matrix @ points

    @classmethod
    def subdivide_bezier(cls, points: np.ndarray, n: int) -> np.ndarray:
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
        if n == 1:
            return points

        subdivision_matrix = cls.subdivision_matrices.get(n, None)
        if subdivision_matrix is None:
            subdivision_matrix = np.empty((3 * n, 3))
            for i in range(n):
                ip1 = i + 1
                nmi = n - i
                nmim1 = nmi - 1
                subdivision_matrix[3 * i : 3 * (i + 1)] = np.array(
                    [
                        [nmi * nmi, 2 * i * nmi, i * i],
                        [nmi * nmim1, i * nmim1 + ip1 * nmi, i * ip1],
                        [nmim1 * nmim1, 2 * ip1 * nmim1, ip1 * ip1],
                    ]
                )
            subdivision_matrix /= n * n
            cls.subdivision_matrices[n] = subdivision_matrix

        return subdivision_matrix @ points

    def make_jagged(self):
        self._internal[1:3] = 0.5 * (self._internal[0:3] + self._internal[2:3])

    def make_smooth(self):
        raise NotImplementedError

    def make_approximately_smooth(self):
        raise NotImplementedError
