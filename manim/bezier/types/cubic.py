from __future__ import annotations

__all__ = ["ManimCubicBezier"]

import sys
import typing
from functools import reduce
from typing import Iterable

import numpy as np

from ...utils.simple_functions import choose
from ...utils.space_ops import cross2d, find_intersection
from ..bezier import ManimBezier


class ManimCubicBezier(ManimBezier):
    degree = 3

    subdivision_matrices = {
        2: np.array(
            [
                [8, 0, 0, 0],
                [4, 4, 0, 0],
                [2, 4, 2, 0],
                [1, 3, 3, 1],
                [1, 3, 3, 1],
                [0, 2, 4, 2],
                [0, 0, 4, 4],
                [0, 0, 0, 8],
            ]
        )
        / 8,
        3: np.array(
            [
                [27, 0, 0, 0],
                [18, 9, 0, 0],
                [12, 12, 3, 0],
                [8, 12, 6, 1],
                [8, 12, 6, 1],
                [4, 12, 9, 2],
                [2, 9, 12, 4],
                [1, 6, 12, 8],
                [1, 6, 12, 8],
                [0, 3, 12, 12],
                [0, 0, 9, 18],
                [0, 0, 0, 27],
            ]
        )
        / 27,
        4: np.array(
            [
                [64, 0, 0, 0],
                [48, 16, 0, 0],
                [36, 24, 4, 0],
                [27, 27, 9, 1],
                [27, 27, 9, 1],
                [18, 30, 14, 2],
                [12, 28, 20, 4],
                [8, 24, 24, 8],
                [8, 24, 24, 8],
                [4, 20, 28, 12],
                [2, 14, 30, 18],
                [1, 9, 27, 27],
                [1, 9, 27, 27],
                [0, 4, 24, 36],
                [0, 0, 16, 48],
                [0, 0, 0, 64],
            ]
        )
        / 64,
    }

    @staticmethod
    def partial_bezier_points(points: np.ndarray, a: float, b: float) -> np.ndarray:
        """Given an array of points which define a Bézier curve, and two numbers 0<=a<b<=1,
        return an array of the same size, which describes the portion of the original Bézier
        curve on the interval [a, b].

        To understand what's going on, see split_bezier for an explanation.

        With that in mind, to find the portion of C0 with t between a and b:
        1. Split C0 at t = a and extract the 2nd curve H1 = [C0(a), Q1(a), L2(a), P3].
        2. Define C0' = H1, and [P0', P1', P2', P3'] = [C0(a), Q1(a), L2(a), P3].
        3. We cannot evaluate C0' at t = b because its range of values for t is different.
        To find the correct value, we need to transform the interval [a, 1] into [0, 1]
        by first subtracting a to get [0, 1-a] and then dividing by 1-a. Thus, our new
        value must be t = (b - a) / (1 - a). Define u = (b-a) / (1-a).
        4. Split C0' at t = u and extract the 1st curve H0' = [P0', L0'(u), Q0'(u), C0'(u)].

        Parameters
        ----------
        points
            set of points defining the bezier curve.
        a
            lower bound of the desired partial bezier curve.
        b
            upper bound of the desired partial bezier curve.

        Returns
        -------
        np.ndarray
            Set of points defining the partial bezier curve.
        """
        arr = np.array(points)  # It is convenient that np.array copies points
        N = arr.shape[0]
        # Border cases
        if a == 1:
            arr[:] = arr[-1]
            return arr
        if b == 0:
            arr[:] = arr[0]
            return arr

        """
        Original algorithm (replace n with N, and points with arr):

        a_to_1 = np.array([bezier(points[i:])(a) for i in range(n)])
        end_prop = (b - a) / (1.0 - a)
        return np.array([bezier(a_to_1[: i + 1])(end_prop) for i in range(n+1)])
        """

        # Current state for an example Bezier curve C0 = [P0, P1, P2, P3]:
        # arr = [P0, P1, P2, P3]
        if a != 0:
            for i in range(1, N):
                # 1st iter: arr = [L0(a), L1(a), L2(a), P3]
                # 2nd iter: arr = [Q0(a), Q1(a), L2(a), P3]
                # 3rd iter: arr = [C0(a), Q1(a), L2(a), P3]
                arr[: N - i] += a * (arr[1 : N - i + 1] - arr[: N - i])

        # For faster calculations we shall define mu = 1 - u = (1 - b) / (1 - a).
        # This is because:
        # L0'(u) = P0' + u(P1' - P0')
        #        = (1-u)P0' + uP1'
        #        = muP0' + (1-mu)P1'
        #        = P1' + mu(P0' - P1)
        # In this way, one can do something similar to the first loop.
        #
        # Current state:
        # arr = [C0(a), Q1(a), L2(a), P3]
        #     = [P0', P1', P2', P3']
        if b != 1:
            if a != 0:
                mu = (1 - b) / (1 - a)
            else:
                mu = 1 - b
            for i in range(1, N):
                # 1st iter: arr = [P0', L0'(u), L1'(u), L2'(u)]
                # 2nd iter: arr = [P0', L0'(u), Q0'(u), Q1'(u)]
                # 3rd iter: arr = [P0', L0'(u), Q0'(u), C0'(u)]
                arr[i:] += mu * (arr[i - 1 : -1] - arr[i:])

        return arr

    @staticmethod
    def split_bezier(points: np.ndarray, t: float) -> np.ndarray:
        """Split a Bézier curve at argument ``t`` into two curves.

        To understand what's going on, let's break this down with an example: a cubic Bézier.

        Let P0, P1, P2, P3 be the points needed for the curve :math:`C_0 = [P_0, P_1, P_2, P_3]`.
        Define the 3 linear Béziers :math:`L_0, L1, L2` as interpolations of :math:`P_0, P_1, P_2, P_3`:
        :math:`L_0(t) = P_0 + t(P_1 - P_0)`
        :math:`L_1(t) = P_1 + t(P_2 - P_1)`
        :math:`L_2(t) = P_2 + t(P_3 - P_2)`
        Define the 2 quadratic Béziers :math:`Q_0, Q_1` as interpolations of :math:`L_0, L_1, L_2`:
        :math:`Q_0(t) = L_0(t) + t(L_1(t) - L_0(t))`
        :math:`Q_1(t) = L_1(t) + t(L_2(t) - L_1(t))`
        Then :math:`C_0` is the following interpolation of :math:`Q_0` and :math:`Q_1`:
        :math:`C_0(t) = Q_0(t) + t(Q_1(t) - Q_0(t))`

        Evaluating :math:`C_0` at a value :math:`t=s` splits :math:`C_0` into two cubic Béziers :math:`H_0`
        and :math:`H_1`, defined by some of the points we calculated earlier:
        - :math:`H_0 = [P_0, L_0(s), Q_0(s), C_0(s)]`
        - :math:`H_1 = [C_0(s), Q_1(s), L_2(s), P_3]`

        Parameters
        ----------
        points
            The control points of the Bézier curve.

        t
            The ``t``-value at which to split the Bézier curve.

        Returns
        -------
            The two Bézier curves as a list of tuples.
        """
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2 = t * t
        t3 = t2 * t
        two_mt_t = 2 * mt * t
        three_mt2_t = 3 * mt2 * t
        three_mt_t2 = 3 * mt * t2

        split_matrix = np.array(
            [
                [1, 0, 0, 0],
                [mt, t, 0, 0],
                [mt2, two_mt_t, t2, 0],
                [mt3, three_mt2_t, three_mt_t2, t3],
                [mt3, three_mt2_t, three_mt_t2, t3],
                [0, mt2, two_mt_t, t2],
                [0, 0, mt, t],
                [0, 0, 0, 1],
            ]
        )

        return split_matrix @ points

    @classmethod
    def subdivide_bezier(cls, points: np.ndarray, n: int) -> np.ndarray:
        """Subdivide a Bézier curve into ``n`` subcurves which have the same shape.

        The points at which the curve is split are located at the
        arguments :math:`t = i/n` for :math:`i = 1, ..., n-1`.

        To see an explanation, see split_bezier.

        Parameters
        ----------
        points
            The control points of the Bézier curve.

        n
            The number of curves to subdivide the Bézier curve into

        Returns
        -------
            The new points for the Bézier curve.

        .. image:: /_static/bezier_subdivision_example.png

        """
        if n == 1:
            return points

        subdivision_matrix = cls.subdivision_matrices.get(n, None)
        if subdivision_matrix is None:
            subdivision_matrix = np.empty((4 * n, 4))
            for i in range(n):
                i2 = i * i
                i3 = i2 * i
                ip1 = i + 1
                ip12 = ip1 * ip1
                ip13 = ip12 * ip1
                nmi = n - i
                nmi2 = nmi * nmi
                nmi3 = nmi2 * nmi
                nmim1 = nmi - 1
                nmim12 = nmi * nmi
                nmim13 = nmi2 * nmi

                subdivision_matrix[4 * i : 4 * (i + 1)] = np.array(
                    [
                        [nmi3, 3 * nmi2 * i, 3 * nmi * i2, i3],
                        [
                            nmi2 * nmim1,
                            2 * nmi * nmim1 * i + nmi2 * ip1,
                            nmim1 * i2 + 2 * nmi * i * ip1,
                            i2 * ip1,
                        ],
                        [
                            nmi * nmim12,
                            nmim12 * i + 2 * nmi * nmim1 * ip1,
                            2 * nmim1 * i * ip1 + nmi * ip12,
                            i * ip12,
                        ],
                        [nmim13, 3 * nmim12 * ip1, 3 * nmim1 * ip12, ip12],
                    ]
                )
            subdivision_matrix /= n * n * n
            cls.subdivision_matrices[n] = subdivision_matrix

        return subdivision_matrix @ points

    def make_jagged(self):
        self._internal[1:4] = (self._internal[0:4] + 2 * self._internal[3:4]) / 3
        self._internal[2:4] = (2 * self._internal[0:4] + self._internal[3:4]) / 3
        return self

    def make_smooth(self):
        h1, h2 = self._smooth_cubic(self._internal)
        self._internal[1:4] = h1
        self._internal[2:4] = h2

    def make_approximately_smooth(self):
        raise NotImplementedError

    @classmethod
    def get_approx_smooth_handle_points(
        cls,
        anchors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @classmethod
    def get_smooth_handle_points(
        cls,
        anchors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return cls._smooth_cubic(anchors)
