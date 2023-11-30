"""Utility functions related to Bézier curves."""

from __future__ import annotations

from manim.typing import (
    BezierPoints,
    ColVector,
    MatrixMN,
    Point3D,
    Point3D_Array,
    PointDType,
    QuadraticBezierPoints,
    QuadraticBezierPoints_Array,
)

__all__ = [
    "bezier",
    "partial_bezier_points",
    "split_bezier",
    "subdivide_bezier",
    "bezier_remap",
    "partial_quadratic_bezier_points",
    "split_quadratic_bezier",
    "subdivide_quadratic_bezier",
    "quadratic_bezier_remap",
    "interpolate",
    "integer_interpolate",
    "mid",
    "inverse_interpolate",
    "match_interpolate",
    "get_smooth_handle_points",
    "get_smooth_cubic_bezier_handle_points",
    "get_smooth_cubic_bezier_handle_points_for_closed_curve",
    "get_smooth_cubic_bezier_handle_points_for_open_curve",
    "diag_to_matrix",
    "is_closed",
    "proportions_along_bezier_curve_for_point",
    "point_lies_on_bezier",
]


from functools import reduce
from typing import Any, Callable, Sequence, overload

import numpy as np
import numpy.typing as npt
from scipy import linalg

from ..utils.simple_functions import choose
from ..utils.space_ops import cross2d, find_intersection


def bezier(
    points: BezierPoints | BezierPoints_Array,
) -> Callable[[float | ColVector], Point3D | Point3D_Array]:
    """Classic implementation of a Bézier curve.

    Parameters
    ----------
    points
        Points defining the desired Bézier curve.

    Returns
    -------
    Callable[[float | ColVector], Point3D | Point3D_Array]
        Function describing the Bézier curve.
        You can either pass a single `t` value between 0 and 1 to get the corresponding
        point on the curve, or an `(N, 1)` column vector of `t` values to get an array
        of points from the curve evaluated at each one of the values.
    """
    P = np.asarray(points)
    n = P.shape[0] - 1

    if n == 0:

        def zero_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            return np.ones_like(t) * P[0]

        return zero_bezier

    if n == 1:

        def linear_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            return P[0] + t * (P[1] - P[0])

        return linear_bezier

    if n == 2:

        def quadratic_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            t2 = t * t
            mt = 1 - t
            mt2 = mt * mt
            return mt2 * P[0] + 2 * t * mt * P[1] + t2 * P[2]

        return quadratic_bezier

    if n == 3:

        def cubic_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            return mt3 * P[0] + 3 * t * mt2 * P[1] + 3 * t2 * mt * P[2] + t3 * P[3]

        return cubic_bezier

    def nth_grade_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
        B = P.copy()
        for i in range(n):
            # After the i-th iteration (i in [0, ..., n-1]) there are (n-i)
            # Bezier curves of grade (i+1) stored in the first n-i slots of B
            B[: n - i] += t * (B[1 : n - i + 1] - B[: n - i])
        # In the end, there shall be a single Bezier curve of grade n
        # stored in the first slot of B
        return B[0]

    return nth_grade_bezier


# TODO: Deprecate and only use partial_bezier_points for handling everything?
def partial_quadratic_bezier_points(
    points: QuadraticBezierPoints, a: float, b: float
) -> list[Point3D]:  # should've been QuadraticBezierPoints
    """Shortened version of partial_bezier_points just for quadratics,
    since this is called a fair amount.

    To understand the mathematics behind splitting curves, see split_bezier.

    Parameters
    ----------
    points
        An array of points defining the quadratic Bézier curve.
    a
        The lower bound of the desired partial quadratic Bézier curve.
    b
        The upper bound of the desired partial quadratic Bézier curve.

    Returns
    -------
    list[Point3D]
        A list of the 3 control points defining the partial quadratic Bézier curve.
    """
    # TODO: this is converted to a list because the current implementation in
    # OpenGLVMobject.insert_n_curves_to_point_list does a list concatenation with +=.
    # Using an ndarray breaks many test cases. This should probably change.
    return list(partial_bezier_points(points, a, b))


# TODO: Deprecate and only use split_bezier for handling everything?
def split_quadratic_bezier(points: QuadraticBezierPoints, t: float) -> Point3D_Array:
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
    Point3D_Array
        An array containing the 6 control points defining the two Bézier curves.
    """
    return split_bezier(points, t)


# TODO: Deprecate and only use subdivide_bezier for handling everything?
def subdivide_quadratic_bezier(points: QuadraticBezierPoints, n: int) -> Point3D_Array:
    """Subdivide a quadratic Bézier curve into ``n`` subcurves which have the same shape.

    The points at which the curve is split are located at the
    arguments :math:`t = i/n` for :math:`i = 1, ..., n-1`.

    To understand the mathematics behind splitting Béziers, see split_bezier.

    Parameters
    ----------
    points
        The control points of the Bézier curve in form ``[a1, h1, b1]``

    n
        The number of curves to subdivide the Bézier curve into

    Returns
    -------
    Point3D_Array
        An array containing the :math:`3n` control points defining the new ``n`` subcurves.

    .. image:: /_static/bezier_subdivision_example.png

    """
    return subdivide_bezier(points, n)


# TODO: Deprecate and only use bezier_remap for handling everything?
def quadratic_bezier_remap(
    triplets: QuadraticBezierPoints_Array, new_number_of_curves: int
) -> QuadraticBezierPoints_Array:
    """Remaps the number of curves to a higher amount by splitting bezier curves

    Parameters
    ----------
    triplets
        The triplets of the quadratic bezier curves to be remapped shape(n, 3, 3)

    new_number_of_curves
        The number of curves that the output will contain. This needs to be higher than the current number.

    Returns
    -------
    QuadraticBezierPoints_Array
        The new triplets for the quadratic bezier curves.
    """

    return bezier_remap(triplets, new_number_of_curves)


def partial_bezier_points(points: BezierPoints, a: float, b: float) -> BezierPoints:
    """Given an array of points which define a Bézier curve, and two numbers 0<=a<b<=1,
    return an array of the same size, which describes the portion of the original Bézier
    curve on the interval [a, b].

    To understand the mathematics behind splitting curves, see split_bezier for an explanation.

    To find the portion of a curve with t between a and b:
    1. Split the curve at t = a and extract its 2nd subcurve.
    2. We cannot evaluate the new subcurve at t = b because its range of values for t is different.
       To find the correct value, we need to transform the interval [a, 1] into [0, 1]
       by first subtracting a to get [0, 1-a] and then dividing by 1-a. Thus, our new
       value must be t = (b - a) / (1 - a). Define u = (b-a) / (1-a).
    3. Split the subcurve at t = u and extract its 1st subcurve.

    The final portion is a linear combination of points and thus the process can be
    summarized as a linear transformation by some matrix in terms of a and b. This matrix
    is given explicitly in the 2nd and 3rd degree cases, which are often used in Manim.

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
    BezierPoints
        An array containing the control points defining the partial Bézier curve.
    """
    # Border cases
    if a == 1:
        arr = np.array(points)
        arr[:] = arr[-1]
        return arr
    if b == 0:
        arr = np.array(points)
        arr[:] = arr[0]
        return arr

    points = np.asarray(points)
    degree = points.shape[0] - 1

    if degree == 3:
        ma, mb = 1 - a, 1 - b
        a2, b2, ma2, mb2 = a * a, b * b, ma * ma, mb * mb
        a3, b3, ma3, mb3 = a2 * a, b2 * b, ma2 * ma, mb2 * mb

        split_matrix = np.array(
            [
                [ma3, 3 * ma2 * a, 3 * ma * a2, a3],
                [ma2 * mb, 2 * ma * a * mb + ma2 * b, a2 * mb + 2 * ma * a * b, a2 * b],
                [ma * mb2, a * mb2 + 2 * ma * mb * b, 2 * a * mb * b + ma * b2, a * b2],
                [mb3, 3 * mb2 * b, 3 * mb * b2, b3],
            ]
        )
        return split_matrix @ points

    if degree == 2:
        ma, mb = 1 - a, 1 - b

        split_matrix = np.array(
            [
                [ma * ma, 2 * a * ma, a * a],
                [ma * mb, a * mb + ma * b, a * b],
                [mb * mb, 2 * b * mb, b * b],
            ]
        )
        return split_matrix @ points

    if degree == 1:
        direction = points[1] - points[0]
        return np.array(
            [
                points[0] + a * direction,
                points[0] + b * direction,
            ]
        )

    if degree == 0:
        return points

    # Fallback case for nth degree Béziers
    # It is convenient that np.array copies points
    arr = np.array(points)
    N = arr.shape[0]

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
        mu = (1 - b) / (1 - a)
        for i in range(1, N):
            # 1st iter: arr = [P0', L0'(u), L1'(u), L2'(u)]
            # 2nd iter: arr = [P0', L0'(u), Q0'(u), Q1'(u)]
            # 3rd iter: arr = [P0', L0'(u), Q0'(u), C0'(u)]
            arr[i:] += mu * (arr[i - 1 : -1] - arr[i:])

    return arr


def split_bezier(points: BezierPoints, t: float) -> Point3D_Array:
    """Split a Bézier curve at argument ``t`` into two curves.

    To understand what's going on, let's break this down with an example: a cubic Bézier.

    Let :math:`P_0, P_1, P_2, P_3` be the points needed for the curve :math:`C_0 = [P_0, P_1, P_2, P_3]`.
    Define the 3 linear Béziers :math:`L_0, L_1, L_2` as interpolations of :math:`P_0, P_1, P_2, P_3`:
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

    As the resulting curves are obtained from linear combinations of ``points``, everything can
    be encoded into a matrix for efficiency.

    Parameters
    ----------
    points
        The control points of the Bézier curve.

    t
        The ``t``-value at which to split the Bézier curve.

    Returns
    -------
    Point3D_Array
        An array containing the control points defining the two Bézier curves.
    """

    points = np.asarray(points)
    N, dim = points.shape
    degree = N - 1

    if degree == 3:
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

    if degree == 2:
        mt = 1 - t
        mt2 = mt * mt
        t2 = t * t
        two_tmt = 2 * t * mt

        split_matrix = np.array(
            [
                [1, 0, 0],
                [mt, t, 0],
                [mt2, two_tmt, t2],
                [mt2, two_tmt, t2],
                [0, mt, t],
                [0, 0, 1],
            ]
        )

        return split_matrix @ points

    if degree == 1:
        middle = points[0] + t * (points[1] - points[0])
        return np.array([points[0], middle, middle, points[1]])

    if degree == 0:
        return np.array([points[0], points[0]])

    # Fallback case for nth degree Béziers
    arr = np.empty((2, N, dim))
    arr[1] = points
    arr[0, 0] = points[0]

    # Example for a cubic Bezier
    # arr[0] = [P0 .. .. ..]
    # arr[1] = [P0 P1 P2 P3]
    for i in range(1, N):
        # 1st iter: arr[1] = [L0 L1 L2 P3]
        # 2nd iter: arr[1] = [Q0 Q1 L2 P3]
        # 3rd iter: arr[1] = [C0 Q1 L2 P3]
        arr[1, : N - i] += t * (arr[1, 1 : N - i + 1] - arr[1, : N - i])
        # 1st iter: arr[0] = [P0 L0 .. ..]
        # 2nd iter: arr[0] = [P0 L0 Q0 ..]
        # 3rd iter: arr[0] = [P0 L0 Q0 C0]
        arr[0, i] = arr[1, 0]

    return arr.reshape(2 * N, dim)


CUBIC_SUBDIVISION_MATRICES = {
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

QUADRATIC_SUBDIVISION_MATRICES = {
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


def subdivide_bezier(points: BezierPoints, n_divisions: int) -> Point3D_Array:
    """Subdivide a Bézier curve into ``n`` subcurves which have the same shape.

    The points at which the curve is split are located at the
    arguments :math:`t = i/n` for :math:`i = 1, ..., n-1`.

    To understand the mathematics behind splitting Béziers, see split_bezier.

    The resulting subcurves can be expressed as linear combinations of
    ``points``, which can be encoded in a single matrix that is precalculated
    for 2nd and 3rd degree Bézier curves.

    Parameters
    ----------
    points
        The control points of the Bézier curve.

    n
        The number of curves to subdivide the Bézier curve into

    Returns
    -------
    Point3D_Array
        An array containing the points defining the new ``n`` subcurves.

    .. image:: /_static/bezier_subdivision_example.png

    """
    if n_divisions == 1:
        return points

    points = np.asarray(points)
    N, dim = points.shape
    degree = N - 1

    if degree == 3:
        subdivision_matrix = CUBIC_SUBDIVISION_MATRICES.get(n_divisions, None)
        if subdivision_matrix is None:
            subdivision_matrix = np.empty((4 * n_divisions, 4))
            for i in range(n_divisions):
                i2 = i * i
                i3 = i2 * i
                ip1 = i + 1
                ip12 = ip1 * ip1
                ip13 = ip12 * ip1
                nmi = n_divisions - i
                nmi2 = nmi * nmi
                nmi3 = nmi2 * nmi
                nmim1 = nmi - 1
                nmim12 = nmi * nmi
                nmim13 = nmi2 * nmi

                subdivision_matrix[4 * i : 4 * (i + 1)] = np.array(
                    [
                        [
                            nmi3,
                            3 * nmi2 * i,
                            3 * nmi * i2,
                            i3,
                        ],
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
                        [
                            nmim13,
                            3 * nmim12 * ip1,
                            3 * nmim1 * ip12,
                            ip13,
                        ],
                    ]
                )
            subdivision_matrix /= n_divisions * n_divisions * n_divisions
            CUBIC_SUBDIVISION_MATRICES[n_divisions] = subdivision_matrix

        return subdivision_matrix @ points

    if degree == 2:
        subdivision_matrix = QUADRATIC_SUBDIVISION_MATRICES.get(n_divisions, None)
        if subdivision_matrix is None:
            subdivision_matrix = np.empty((3 * n_divisions, 3))
            for i in range(n_divisions):
                ip1 = i + 1
                nmi = n_divisions - i
                nmim1 = nmi - 1
                subdivision_matrix[3 * i : 3 * (i + 1)] = np.array(
                    [
                        [nmi * nmi, 2 * i * nmi, i * i],
                        [nmi * nmim1, i * nmim1 + ip1 * nmi, i * ip1],
                        [nmim1 * nmim1, 2 * ip1 * nmim1, ip1 * ip1],
                    ]
                )
            subdivision_matrix /= n_divisions * n_divisions
            QUADRATIC_SUBDIVISION_MATRICES[n_divisions] = subdivision_matrix

        return subdivision_matrix @ points

    if degree == 1:
        return points[0] + np.linspace(0, 1, n_divisions + 1).reshape(-1, 1) * (
            points[1] - points[0]
        )

    if degree == 0:
        arr = np.empty((n_divisions + 1, dim))
        arr[:] = points[0]
        return arr

    # Fallback case for an nth degree Bézier: successive splitting
    beziers = np.empty((n_divisions, N, dim))
    beziers[-1] = points
    for curve_num in range(n_divisions - 1, 0, -1):
        curr = beziers[curve_num]
        prev = beziers[curve_num - 1]
        prev[0] = curr[0]
        # Current state for an example cubic Bézier curve:
        # prev = [P0 .. .. ..]
        # curr = [P0 P1 P2 P3]
        for i in range(1, N):
            a = (n_divisions - curve_num - 1) / (n_divisions - curve_num)
            # 1st iter: curr = [L0 L1 L2 P3]
            # 2nd iter: curr = [Q0 Q1 L2 P3]
            # 3rd iter: curr = [C0 Q1 L2 P3]
            curr[: N - i] += a * (curr[1 : N - i + 1] - curr[: N - i])
            # 1st iter: prev = [P0 L0 .. ..]
            # 2nd iter: prev = [P0 L0 Q0 ..]
            # 3rd iter: prev = [P0 L0 Q0 C0]
            prev[i] = curr[0]

    return beziers.reshape(n_divisions * N, dim)


def bezier_remap(
    bezier_tuples: BezierPoints_Array,
    new_number_of_curves: int,
) -> BezierPoints_Array:
    """Subdivides each curve in ``bezier_tuples`` into as many parts as necessary, until the final number of
    curves reaches a desired amount, ``new_number_of_curves``.

    Parameters
    ----------
    bezier_tuples
        An array of n Bézier curves to be remapped. The shape of this array must be (current_num_curves, degree+1, dimension).
    new_number_of_curves
        The number of curves that the output will contain. This needs to be higher than the current number.

    Returns
    -------
    BezierPoints_Array
        The new Bézier curves after the remap.
    """
    bezier_tuples = np.asarray(bezier_tuples)
    current_number_of_curves, nppc, dim = bezier_tuples.shape
    # This is an array with values ranging from 0
    # up to curr_num_curves,  with repeats such that
    # it's total length is target_num_curves.  For example,
    # with curr_num_curves = 10, target_num_curves = 15, this
    # would be [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9].
    repeat_indices = (
        np.arange(new_number_of_curves, dtype="i") * current_number_of_curves
    ) // new_number_of_curves

    # If the nth term of this list is k, it means
    # that the nth curve of our path should be split
    # into k pieces.
    # In the above example our array had the following elements
    # [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9]
    # We have two 0s, one 1, two 2s and so on.
    # The split factors array would hence be:
    # [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
    split_factors = np.zeros(current_number_of_curves, dtype="i")
    np.add.at(split_factors, repeat_indices, 1)

    new_tuples = np.empty((new_number_of_curves, nppc, dim))
    index = 0
    for curve, sf in zip(bezier_tuples, split_factors):
        new_tuples[index : index + sf] = subdivide_bezier(curve, sf).reshape(
            sf, nppc, dim
        )
        index += sf

    return new_tuples


# Linear interpolation variants


@overload
def interpolate(start: float, end: float, alpha: float) -> float:
    ...


@overload
def interpolate(start: Point3D, end: Point3D, alpha: float) -> Point3D:
    ...


def interpolate(
    start: int | float | Point3D, end: int | float | Point3D, alpha: float | Point3D
) -> float | Point3D:
    return start + alpha * (end - start)


def integer_interpolate(
    start: float,
    end: float,
    alpha: float,
) -> tuple[int, float]:
    """
    This is a variant of interpolate that returns an integer and the residual

    Parameters
    ----------
    start
        The start of the range
    end
        The end of the range
    alpha
        a float between 0 and 1.

    Returns
    -------
    tuple[int, float]
        This returns an integer between start and end (inclusive) representing
        appropriate interpolation between them, along with a
        "residue" representing a new proportion between the
        returned integer and the next one of the
        list.

    Example
    -------

    .. code-block:: pycon

        >>> integer, residue = integer_interpolate(start=0, end=10, alpha=0.46)
        >>> np.allclose((integer, residue), (4, 0.6))
        True
    """
    if alpha >= 1:
        return (int(end - 1), 1.0)
    if alpha <= 0:
        return (int(start), 0)
    value = int(interpolate(start, end, alpha))
    residue = ((end - start) * alpha) % 1
    return (value, residue)


@overload
def mid(start: float, end: float) -> float:
    ...


@overload
def mid(start: Point3D, end: Point3D) -> Point3D:
    ...


def mid(start: float | Point3D, end: float | Point3D) -> float | Point3D:
    """Returns the midpoint between two values.

    Parameters
    ----------
    start
        The first value
    end
        The second value

    Returns
    -------
        The midpoint between the two values
    """
    return (start + end) / 2.0


@overload
def inverse_interpolate(start: float, end: float, value: float) -> float:
    ...


@overload
def inverse_interpolate(start: float, end: float, value: Point3D) -> Point3D:
    ...


@overload
def inverse_interpolate(start: Point3D, end: Point3D, value: Point3D) -> Point3D:
    ...


def inverse_interpolate(
    start: float | Point3D, end: float | Point3D, value: float | Point3D
) -> float | Point3D:
    """Perform inverse interpolation to determine the alpha
    values that would produce the specified ``value``
    given the ``start`` and ``end`` values or points.

    Parameters
    ----------
    start
        The start value or point of the interpolation.
    end
        The end value or point of the interpolation.
    value
        The value or point for which the alpha value
        should be determined.

    Returns
    -------
        The alpha values producing the given input
        when interpolating between ``start`` and ``end``.

    Example
    -------

    .. code-block:: pycon

        >>> inverse_interpolate(start=2, end=6, value=4)
        0.5

        >>> start = np.array([1, 2, 1])
        >>> end = np.array([7, 8, 11])
        >>> value = np.array([4, 5, 5])
        >>> inverse_interpolate(start, end, value)
        array([0.5, 0.5, 0.4])
    """
    return np.true_divide(value - start, end - start)


@overload
def match_interpolate(
    new_start: float,
    new_end: float,
    old_start: float,
    old_end: float,
    old_value: float,
) -> float:
    ...


@overload
def match_interpolate(
    new_start: float,
    new_end: float,
    old_start: float,
    old_end: float,
    old_value: Point3D,
) -> Point3D:
    ...


def match_interpolate(
    new_start: float,
    new_end: float,
    old_start: float,
    old_end: float,
    old_value: float | Point3D,
) -> float | Point3D:
    """Interpolate a value from an old range to a new range.

    Parameters
    ----------
    new_start
        The start of the new range.
    new_end
        The end of the new range.
    old_start
        The start of the old range.
    old_end
        The end of the old range.
    old_value
        The value within the old range whose corresponding
        value in the new range (with the same alpha value)
        is desired.

    Returns
    -------
        The interpolated value within the new range.

    Examples
    --------
    >>> match_interpolate(0, 100, 10, 20, 15)
    50.0
    """
    old_alpha = inverse_interpolate(old_start, old_end, old_value)
    return interpolate(
        new_start,
        new_end,
        old_alpha,  # type: ignore
    )


# Figuring out which Bezier curves most smoothly connect a sequence of points
# TODO: Include quadratic splines here
def get_smooth_handle_points(
    anchors: Point3D_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    """Given an array of anchors for a spline (array of connected cubic
    Bézier curves), compute the handles for every curve, so that the resulting
    spline is smooth.

    Currently this function only redirects to
    :func:`get_smooth_cubic_bezier_handle_points`, because the algorithm is
    only implemented for cubic splines. In the future, this should also include
    at least the case for quadratic splines.

    Parameters
    ----------
    anchors
        Anchors of a cubic spline.

    Returns
    -------
    tuple[Point3D_Array, Point3D_Array]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the cubic spline, and the other containing the 2nd handles.
    """
    return get_smooth_cubic_bezier_handle_points(anchors)


def get_smooth_cubic_bezier_handle_points(
    anchors: Point3D_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    """Given an array of anchors for a cubic spline (array of connected cubic
    Bézier curves), compute the 1st and 2nd handle for every curve, so that
    the resulting spline is smooth.

    Parameters
    ----------
    anchors
        Anchors of a cubic spline.

    Returns
    -------
    tuple[Point3D_Array, Point3D_Array]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the cubic spline, and the other containing the 2nd handles.
    """
    anchors = np.asarray(anchors)
    n_handles = len(anchors) - 1

    # If there's a single anchor, there's no Bezier curve.
    # Return empty arrays.
    if n_handles == 0:
        dim = anchors.shape[1]
        return np.zeros((0, dim)), np.zeros((0, dim))

    # If there are only two anchors (thus only one pair of handles),
    # they can only be an interpolation of these two anchors with alphas
    # 1/3 and 2/3, which will draw a straight line between the anchors.
    if n_handles == 1:
        return interpolate(anchors[0], anchors[1], np.array([[1 / 3], [2 / 3]]))

    # Handle different cases depending on whether the points form a closed
    # curve or not
    curve_is_closed = is_closed(anchors)
    if curve_is_closed:
        return get_smooth_cubic_bezier_handle_points_for_closed_curve(anchors)
    else:
        return get_smooth_cubic_bezier_handle_points_for_open_curve(anchors)


CP_CLOSED_MEMO = np.array([1 / 3])
UP_CLOSED_MEMO = np.array([1 / 3])


def get_smooth_cubic_bezier_handle_points_for_closed_curve(
    anchors: Point3D_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    """Special case of :func:`get_smooth_cubic_bezier_handle_points`,
    when the `anchors` form a closed loop.

    A system of equations must be solved to get the first handles of
    every Bèzier curve (referred to as H1).
    Then H2 (the second handles) can be obtained separately.
    The equations were obtained from:
    http://www.jacos.nl/jacos_html/spline/theory/theory_2.html

    In general, if there are N+1 anchors there will be N Bezier curves
    and thus N pairs of handles to find. We must solve the following
    system of equations for the 1st handles (example for N = 5):

    [4 1 0 0 1]   [H1[0]]   [4*A[0] + 2*A[1]]
    [1 4 1 0 0]   [H1[1]]   [4*A[1] + 2*A[2]]
    [0 1 4 1 0]   [H1[2]]   [4*A[2] + 2*A[3]]
    [0 0 1 4 1] @ [H1[3]] = [4*A[3] + 2*A[4]]
    [1 0 0 1 4]   [H1[4]]   [4*A[4] + 2*A[5]]

    which will be expressed as M @ H1 = d.
    M is almost a tridiagonal matrix, so we could use Thomas' algorithm:
    see https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    However, M has ones at the opposite corners. A solution to this is
    the first decomposition proposed here, with alpha = 1:
    https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants

    [4 1 0 0 1]   [3 1 0 0 0]   [1 0 0 0 1]
    [1 4 1 0 0]   [1 4 1 0 0]   [0 0 0 0 0]
    [0 1 4 1 0] = [0 1 4 1 0] + [0 0 0 0 0]
    [0 0 1 4 1]   [0 0 1 4 1]   [0 0 0 0 0]
    [1 0 0 1 4]   [0 0 0 1 3]   [1 0 0 0 1]

                  [3 1 0 0 0]   [1]
                  [1 4 1 0 0]   [0]
                = [0 1 4 1 0] + [0] @ [1 0 0 0 1]
                  [0 0 1 4 1]   [0]
                  [0 0 0 1 3]   [1]

    We decompose M = N + u @ v.T, where N is a tridiagonal matrix, and u
    and v are N-D vectors such that u[0]=u[N-1]=v[0]=v[N-1] = 1, and
    u[i] = v[i] = 0 for all i in {1, ..., N-2}.

    Thus:
       M @ H1 = d
    => (N + u @ v.T) @ H1 = d

    If we find a vector q such that N @ q = u:
    => (N + N @ q @ v.T) @ H1 = d
    => N @ (I + q @ v.T) @ H1 = d
    => H1 = (I + q @ v.T)⁻¹ @ N⁻¹ @ d

    According to Sherman-Morrison's formula, which is explained here:
    https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    (I + q @ v.T)⁻¹ = I - 1/(1 + v.T @ q) * (q @ v.T)

    If we find y = N⁻¹ @ d, or in other words, if we solve for y in N @ y = d:
    => H1 = y - 1/(1 + v.T @ q) * (q @ v.T @ y)

    So we must solve for q and y in N @ q = u and N @ y = d.
    As N is now tridiagonal, we shall use Thomas' algorithm.

    Let a = [a[0], a[1], ..., a[N-2]] be the lower diagonal of N-1 elements,
    such that a[0]=a[1]=...=a[N-2] = 1, so this diagonal is filled with ones;
        b = [b[0], b[1], ..., b[N-2], b[N-1]] the main diagonal of N elements,
    such that b[0]=b[N-1] = 3, and b[1]=b[2]=...=b[N-2] = 4;
    and c = [c[0], c[1], ..., c[N-2]] the upper diagonal of N-1 elements,
    such that c[0]=c[1]=...=c[N-2] = 1: this diagonal is also filled with ones.

    If, according to Thomas' algorithm, we define:
    c'[0] = c[0] / b[0]
    c'[i] = c[i] / (b[i] - a[i-1]*c'[i-1]) = 1/(4-c'[i-1]),  i in [1, ..., N-2]
    u'[0] = u[0] / b[0]
    u'[i] = (u[i] - a[i-1]*u'[i-1]) / (b[i] - a[i-1]*c'[i-1]), i in [1, ..., N-1]
    d'[0] = d[0] / b[0]
    d'[i] = (d[i] - a[i-1]*d'[i-1]) / (b[i] - a[i-1]*c'[i-1]), i in [1, ..., N-1]

    Then:
    c'[0]   = 1/3
    c'[i]   = 1 / (4 - c'[i-1]), if i in {1, ..., N-2}
    u'[0]   = 1/3
    u'[i]   = -u'[i-1] / (4 - c'[i-1])
            = -c'[i]*u'[i-1],                                i in [1, ..., N-2]
    u'[N-1] = (1-u'[N-2]) / (3 - c'[N-2])
    d'[0]   = (4*A[0] + 2*A[1]) / 3
    d'[i]   = (4*A[i] + 2*A[i+1] - d'[i-1]) / (4 - c'[i-1])
            = c'[i] * (4*A[i] + 2*A[i+1] - d'[i-1]),         i in [1, ..., N-2]
    d'[N-1] = (4*A[N-1] + 2*A[N] - d'[N-2]) / (3 - c'[N-2])

    Finally, we can do Backward Substitution to find q and y:
    q[N-1] = u'[N-1]
    q[i]   = u'[i] - c'[i]*q[i+1], for i in [N-2, ..., 0]
    y[N-1] = d'[N-1]
    y[i]   = d'[i] - c'[i]*y[i+1], for i in [N-2, ..., 0]

    With those values, we can calculate H1 = y - 1/(1 + v.T @ q) * (q @ v.T @ y).
    Given that v[0]=v[N-1] = 1, and v[1]=v[2]=...=v[N-2] = 0, its dot products
    with q and y are respectively q[0]+q[N-1] and y[0]+y[N-1]. Thus:
    H1 = y - (y[0]+y[N-1]) / (1+q[0]+q[N-1]) * q

    Once we have H1, we can get H2 (the array of second handles) as follows:
    H2[i]   = 2*A[i+1] - H1[i+1], for i in [0, ..., N-2]
    H2[N-1] = 2*A[0]   - H1[0]

    Because the matrix M (and thus N, u and v) always follows the same pattern,
    we can define a memo list for c' and u' to avoid recalculation. We cannot
    memoize d and y, however, because they are always different vectors. We
    cannot make a memo for q either, but we can calculate it faster because u'
    can be memoized.

    Parameters
    ----------
    anchors
        Anchors of a closed cubic spline.

    Returns
    -------
    tuple[Point3D_Array, Point3D_Array]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the closed cubic spline, and the other containing the 2nd handles.
    """
    global CP_CLOSED_MEMO
    global UP_CLOSED_MEMO

    A = np.asarray(anchors)
    N = len(anchors) - 1
    dim = A.shape[1]

    # Calculate cp (c prime) and up (u prime) with help from
    # CP_CLOSED_MEMO and UP_CLOSED_MEMO.
    len_memo = CP_CLOSED_MEMO.size
    if len_memo < N - 1:
        cp = np.empty(N - 1)
        up = np.empty(N - 1)
        cp[:len_memo] = CP_CLOSED_MEMO
        up[:len_memo] = UP_CLOSED_MEMO
        # Forward Substitution 1
        # Calculate up (at the same time we calculate cp).
        for i in range(len_memo, N - 1):
            cp[i] = 1 / (4 - cp[i - 1])
            up[i] = -cp[i] * up[i - 1]
        CP_CLOSED_MEMO = cp
        UP_CLOSED_MEMO = up
    else:
        cp = CP_CLOSED_MEMO[: N - 1]
        up = UP_CLOSED_MEMO[: N - 1]

    # The last element of u' is different
    cp_last_division = 1 / (3 - cp[N - 2])
    up_last = cp_last_division * (1 - up[N - 2])

    # Backward Substitution 1
    # Calculate q.
    q = np.empty((N, dim))
    q[N - 1] = up_last
    for i in range(N - 2, -1, -1):
        q[i] = up[i] - cp[i] * q[i + 1]

    # Forward Substitution 2
    # Calculate dp (d prime).
    dp = np.empty((N, dim))
    aux = 4 * A[:N] + 2 * A[1:]  # Vectorize the sum for efficiency.
    dp[0] = aux[0] / 3
    for i in range(1, N - 1):
        dp[i] = cp[i] * (aux[i] - dp[i - 1])
    dp[N - 1] = cp_last_division * (aux[N - 1] - dp[N - 2])

    # Backward Substitution
    # Calculate y, which is defined as a view of dp for efficiency
    # and semantic convenience at the same time.
    y = dp
    # y[N-1] = dp[N-1] (redundant)
    for i in range(N - 2, -1, -1):
        y[i] = dp[i] - cp[i] * y[i + 1]

    # Calculate H1.
    H1 = y - (y[0] + y[N - 1]) / (1 + q[0] + q[N - 1]) * q

    # Calculate H2.
    H2 = np.empty((N, dim))
    H2[0 : N - 1] = 2 * A[1:N] - H1[1:N]
    H2[N - 1] = 2 * A[N] - H1[0]

    return H1, H2


CP_OPEN_MEMO = np.array([0.5])


def get_smooth_cubic_bezier_handle_points_for_open_curve(
    anchors: Point3D_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    """Special case of :func:`get_smooth_cubic_bezier_handle_points`,
    when the `anchors` do not form a closed loop.

    A system of equations must be solved to get the first handles of
    every Bèzier curve (referred to as `H1`).
    Then `H2` (the second handles) can be obtained separately.
    The equations were obtained from:
    https://www.particleincell.com/2012/bezier-splines/
    http://www.jacos.nl/jacos_html/spline/theory/theory_2.html
    WARNING: the equations in the first webpage have some typos which
    were corrected in the comments.

    In general, if there are :math:`N+1` anchors there will be N Bezier curves
    and thus :math:`N` pairs of handles to find. We must solve the following
    system of equations for the 1st handles (example for :math:`N = 5`):

    [2 1 0 0 0]   [H1[0]]   [  A[0] + 2*A[1]]
    [1 4 1 0 0]   [H1[1]]   [4*A[1] + 2*A[2]]
    [0 1 4 1 0] @ [H1[2]] = [4*A[2] + 2*A[3]]
    [0 0 1 4 1]   [H1[3]]   [4*A[3] + 2*A[4]]
    [0 0 0 2 7]   [H1[4]]   [8*A[4] +   A[5]]

    which will be expressed as `M @ H1 = d`.
    :math:`M` is a tridiagonal matrix, so the system can be solved in :math`O(n)`
    operations. Here we shall use Thomas' algorithm or the tridiagonal matrix
    algorithm. See: <https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm>

    Let `a = [a[0], a[1], ..., a[N-2]]` be the lower diagonal of :math:`N-1` elements,
    such that `a[0]=a[1]=...=a[N-3] = 1`, and `A[N-2] = 2`;
        `b = [b[0], b[1], ..., b[N-2], b[N-1]]` the main diagonal of :math:`N` elements,
    such that `b[0] = 2, b[1]=b[2]=...=b[N-2] = 4`, and `b[N-1] = 7`;
    and `c = [c[0], c[1], ..., c[N-2]]` the upper diagonal of :math:`N-1` elements,
    such that `c[0]=c[1]=...=c[N-2] = 1`: this diagonal is filled with ones.

    If, according to Thomas' algorithm, we define:
    c'[0] = c[0] / b[0]
    c'[i] = c[i] / (b[i] - a[i-1]*c'[i-1]) = 1/(4-c'[i-1]),  i in [1, ..., N-2]
    d'[0] = d[0] / b[0]
    d'[i] = (d[i] - a[i-1]*d'[i-1]) / (b[i] - a[i-1]*c'[i-1]), i in [1, ..., N-1]

    Then:
    c'[0]   = 0.5
    c'[i]   = 1 / (4 - c'[i-1]), if i in {1, ..., N-2}
    d'[0]   = 0.5*A[0] + A[1]
    d'[i]   = (4*A[i] + 2*A[i+1] - d'[i-1]) / (4 - c'[i-1])
            = c'[i] * (4*A[i] + 2*A[i+1] - d'[i-1]),         i in [1, ..., N-2]
    d'[N-1] = (8*A[N-1] + A[N] - 2*d'[N-2]) / (7 - 2*c'[N-2])

    Finally, we can do Backward Substitution to find `H1`:
    H1[N-1] = d'[N-1]
    H1[i]   = d'[i] - c'[i]*H1[i+1], for i in [N-2, ..., 0]

    Once we have `H1`, we can get `H2` (the array of second handles) as follows:
    H2[i]   =   2*A[i+1]     - H1[i+1], for i in [0, ..., N-2]
    H2[N-1] = 0.5*A[N]   + 0.5*H1[N-1]

    As the matrix :math:`M` always follows the same pattern, we can define a memo list
    for :math:`c'` to avoid recalculation. We cannot do the same for :math:`d`, however,
    because it is always a different vector.

    Parameters
    ----------
    anchors
        Anchors of an open cubic spline.

    Returns
    -------
    tuple[Point3D_Array, Point3D_Array]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the open cubic spline, and the other containing the 2nd handles.
    """
    global CP_OPEN_MEMO

    A = np.asarray(anchors)
    N = len(anchors) - 1
    dim = A.shape[1]

    # Calculate cp (c prime) with help from CP_OPEN_MEMO.
    len_memo = CP_OPEN_MEMO.size
    if len_memo < N - 1:
        cp = np.empty(N - 1)
        cp[:len_memo] = CP_OPEN_MEMO
        for i in range(len_memo, N - 1):
            cp[i] = 1 / (4 - cp[i - 1])
        CP_OPEN_MEMO = cp
    else:
        cp = CP_OPEN_MEMO[: N - 1]

    # Calculate dp (d prime).
    dp = np.empty((N, dim))
    dp[0] = 0.5 * A[0] + A[1]
    aux = 4 * A[1 : N - 1] + 2 * A[2:N]  # Vectorize the sum for efficiency.
    for i in range(1, N - 1):
        dp[i] = cp[i] * (aux[i - 1] - dp[i - 1])
    dp[N - 1] = (8 * A[N - 1] + A[N] - 2 * dp[N - 2]) / (7 - 2 * cp[N - 2])

    # Backward Substitution.
    # H1 (array of the first handles) is defined as a view of dp for efficiency
    # and semantic convenience at the same time.
    H1 = dp
    # H1[N-1] = dp[N-1] (redundant)
    for i in range(N - 2, -1, -1):
        H1[i] = dp[i] - cp[i] * H1[i + 1]

    # Calculate H2.
    H2 = np.empty((N, dim))
    H2[0 : N - 1] = 2 * A[1:N] - H1[1:N]
    H2[N - 1] = 0.5 * (A[N] + H1[N - 1])

    return H1, H2


# TODO: because get_smooth_handle_points was rewritten, this function
# is no longer used. Deprecate?
def diag_to_matrix(
    l_and_u: tuple[int, int], diag: npt.NDArray[Any]
) -> npt.NDArray[Any]:
    """
    Converts array whose rows represent diagonal
    entries of a matrix into the matrix itself.
    See `scipy.linalg.solve_banded`.

    Parameters
    ----------
    l_and_u
        Tuple containing `l` (n° of subdiagonals) and `u` (n° of superdiagonals).
    diag
        2D-array containing the diagonals of the matrix.

    Returns
    -------
    npt.NDArray[Any]
        The resulting matrix.
    """
    l, u = l_and_u
    dim = diag.shape[1]
    matrix = np.zeros((dim, dim))
    for i in range(l + u + 1):
        np.fill_diagonal(
            matrix[max(0, i - u) :, max(0, u - i) :],
            diag[i, max(0, u - i) :],
        )
    return matrix


def get_quadratic_approximation_of_cubic(
    a0: Point3D | Point3D_Array,
    h0: Point3D | Point3D_Array,
    h1: Point3D | Point3D_Array,
    a1: Point3D | Point3D_Array,
) -> Point3D_Array:
    """If a0, h0, h1 and a1 are `(3,)`-ndarrays representing control points for a
    cubic Bézier curve, returns a `(6, 3)-ndarray of 6 control points
    [a'0, h', a'1, a''0, h'', a''1] for 2 quadratic Bézier curves approximating it.

    If a0, h0, h1 and a1 are `(m, 3)`-ndarrays of `m` control points for `m`
    cubic Bézier curves, returns instead a `(6*m, 3)`-ndarray of `6*m` control
    points, where each one of the `m` groups of 6 control points defines the 2
    quadratic curves approximating the respective cubic curve.

    Parameters
    ----------
    a0
        A (3,) or (m, 3)-ndarray of the start anchor(s) of the cubic Bézier curve(s).
    h0
        A (3,) or (m, 3)-ndarray of the first handle(s) of the cubic Bézier curve(s).
    h1
        A (3,) or (m, 3)-ndarray of the second handle(s) of the cubic Bézier curve(s).
    a1
        A (3,) or (m, 3)-ndarray of the end anchor(s) of the cubic Bézier curve(s).

    Returns
    -------
    Point3D_Array
        A `(6*m, 3)`-ndarray, where each one of the `m` groups of
        consecutive 6 points defines the 2 quadratic curves which
        approximate the respective cubic curve.
    """
    # If a0 is a Point3D, it's converted into a Point3D_Array of a single point:
    # its shape is now (1, 3).
    # If it was already a Point3D_Array of m points, it keeps its (m, 3) shape.
    # Same with the other parameters.
    a0 = np.array(a0, ndmin=2)
    h0 = np.array(h0, ndmin=2)
    h1 = np.array(h1, ndmin=2)
    a1 = np.array(a1, ndmin=2)
    # Tangent vectors at the start and end.
    T0 = h0 - a0
    T1 = a1 - h1

    # Search for inflection points. This happens when the acceleration (2nd derivative)
    # is either zero or perpendicular to the velocity (1st derivative), captured
    # in the cross product equation C'(t) x C''(t) = 0.
    # If no inflection points are found, use the midpoint as the split point instead.
    # Based on https://pomax.github.io/bezierinfo/#inflections
    t_split = np.full(a0.shape[0], 0.5)

    # Let C(t) be a cubic Bézier curve defined by [a0, h0, h1, a1].
    # The below variables p, q, r allow for expressing the curve in a
    # polynomial form convenient for derivatives:
    # C(t) = a0 + 3tp + 3t²q + t³r
    p = h0 - a0
    q = h1 - 2 * h0 + a0
    r = a1 - 3 * h1 + 3 * h0 - a0

    # Velocity:      C'(t) = 3p + 6tq + 3t²r
    # Acceleration: C''(t) = 6q + 6tr
    #       C'(t) x C''(t) = 18 (t²(qxr) + t(pxr) + (pxq)) = 0
    # Define a = (qxr), b = (pxr) and c = (pxq).
    # If C(t) is a 2D curve, then a, b and c are real numbers and
    # this is a simple quadratic equation.
    # However, if it's a 3D curve, then a, b and c are 3D vectors
    # and this would require solving 3 quadratic equations.
    # TODO: this simplifies by considering the 2D case. It might fail with 3D curves!
    a = cross2d(q, r)
    b = cross2d(p, r)
    c = cross2d(p, q)

    # Case a == 0: degenerate 1st degree equation bt + c = 0 => t = -c/b
    is_quadratic = a != 0
    is_linear = (~is_quadratic) & (b != 0)
    t_split[is_linear] = -c[is_linear] / b[is_linear]
    # Note: If a == 0 and b == 0, there are 0 or infinite solutions.
    # Thus there are no inflection points. Just leave as is: t = 0.5

    # Case a != 0: 2nd degree equation at² + bt + c = 0
    # => t = -(b/2a) +- sqrt((b/2a)² - c/a)
    # Define u = b/2a and v = c/a, so that t = -u +- sqrt(u² - v).
    u = 0.5 * b[is_quadratic] / a[is_quadratic]
    v = c[is_quadratic] / a[is_quadratic]
    radical = u * u - v
    is_real = radical >= 0
    sqrt_radical = np.sqrt(radical[is_real])

    t_minus = u[is_real] - sqrt_radical
    t_plus = u[is_real] + sqrt_radical
    is_t_minus_valid = (t_minus > 0) & (t_minus < 1)
    is_t_plus_valid = (t_plus > 0) & (t_plus < 1)

    t_split[is_quadratic][is_real][is_t_minus_valid] = t_minus[is_t_minus_valid]
    t_split[is_quadratic][is_real][is_t_plus_valid] = t_plus[is_t_plus_valid]

    # Compute bezier point and tangent at the chosen value of t (these are vectorized)
    t_split = t_split.reshape(-1, 1)
    split_point = bezier([a0, h0, h1, a1])(t_split)  # type: ignore
    tangent_at_split = bezier([h0 - a0, h1 - h0, a1 - h1])(t_split)  # type: ignore

    # Intersection between tangent lines at end points
    # and tangent in the middle
    i0 = find_intersection(a0, T0, split_point, tangent_at_split)
    i1 = find_intersection(a1, T1, split_point, tangent_at_split)

    m, n = np.shape(a0)
    result = np.empty((6 * m, n))
    result[0::6] = a0
    result[1::6] = i0
    result[2::6] = split_point
    result[3::6] = split_point
    result[4::6] = i1
    result[5::6] = a1
    return result


def is_closed(points: Point3D_Array) -> bool:
    """Returns True if the curve given by the points is closed, by checking if its
    first and last points are close to each other.

    This function reimplements np.allclose (without a relative tolerance rtol),
    because repeated calling of np.allclose for only 2 points is inefficient.

    Parameters
    ----------
    points
        An array of points.

    Returns
    -------
    bool
        Whether the first and last points of the array are close enough or not
        to be considered the same, thus considering the defined curve as closed.
    """
    start, end = points[0], points[-1]
    atol = 1e-8
    if abs(end[0] - start[0]) > atol:
        return False
    if abs(end[1] - start[1]) > atol:
        return False
    if abs(end[2] - start[2]) > atol:
        return False
    return True


def proportions_along_bezier_curve_for_point(
    point: Point3D,
    control_points: BezierPoints,
    round_to: float = 1e-6,
) -> npt.NDArray[Any]:
    """Obtains the proportion along the bezier curve corresponding to a given point
    given the bezier curve's control points.

    The bezier polynomial is constructed using the coordinates of the given point
    as well as the bezier curve's control points. On solving the polynomial for each dimension,
    if there are roots common to every dimension, those roots give the proportion along the
    curve the point is at. If there are no real roots, the point does not lie on the curve.

    Parameters
    ----------
    point
        The Cartesian Coordinates of the point whose parameter
        should be obtained.
    control_points
        The Cartesian Coordinates of the ordered control
        points of the bezier curve on which the point may
        or may not lie.
    round_to
        A float whose number of decimal places all values
        such as coordinates of points will be rounded.

    Returns
    -------
        np.ndarray[float]
            List containing possible parameters (the proportions along the bezier curve)
            for the given point on the given bezier curve.
            This usually only contains one or zero elements, but if the
            point is, say, at the beginning/end of a closed loop, may return
            a list with more than 1 value, corresponding to the beginning and
            end etc. of the loop.

    Raises
    ------
    :class:`ValueError`
        When ``point`` and the control points have different shapes.
    """
    # Method taken from
    # http://polymathprogrammer.com/2012/04/03/does-point-lie-on-bezier-curve/

    if not all(np.shape(point) == np.shape(c_p) for c_p in control_points):
        raise ValueError(
            f"Point {point} and Control Points {control_points} have different shapes.",
        )

    control_points = np.array(control_points)
    n = len(control_points) - 1

    roots = []
    for dim, coord in enumerate(point):
        control_coords = control_points[:, dim]
        terms = []
        for term_power in range(n, -1, -1):
            outercoeff = choose(n, term_power)
            term = []
            sign = 1
            for subterm_num in range(term_power, -1, -1):
                innercoeff = choose(term_power, subterm_num) * sign
                subterm = innercoeff * control_coords[subterm_num]
                if term_power == 0:
                    subterm -= coord
                term.append(subterm)
                sign *= -1
            terms.append(outercoeff * sum(np.array(term)))
        if all(term == 0 for term in terms):
            # Then both Bezier curve and Point lie on the same plane.
            # Roots will be none, but in this specific instance, we don't need to consider that.
            continue
        bezier_polynom = np.polynomial.Polynomial(terms[::-1])
        polynom_roots = bezier_polynom.roots()  # type: ignore
        if len(polynom_roots) > 0:
            polynom_roots = np.around(polynom_roots, int(np.log10(1 / round_to)))
        roots.append(polynom_roots)

    roots = [[root for root in rootlist if root.imag == 0] for rootlist in roots]
    # Get common roots
    # arg-type: ignore
    roots = reduce(np.intersect1d, roots)  # type: ignore
    result = np.asarray([r.real for r in roots if 0 <= r.real <= 1])
    return result


def point_lies_on_bezier(
    point: Point3D,
    control_points: BezierPoints,
    round_to: float = 1e-6,
) -> bool:
    """Checks if a given point lies on the bezier curves with the given control points.

    This is done by solving the bezier polynomial with the point as the constant term; if
    any real roots exist, the point lies on the bezier curve.

    Parameters
    ----------
    point
        The Cartesian Coordinates of the point to check.
    control_points
        The Cartesian Coordinates of the ordered control
        points of the bezier curve on which the point may
        or may not lie.
    round_to
        A float whose number of decimal places all values
        such as coordinates of points will be rounded.

    Returns
    -------
    bool
        Whether the point lies on the curve.
    """

    roots = proportions_along_bezier_curve_for_point(point, control_points, round_to)

    return len(roots) > 0
