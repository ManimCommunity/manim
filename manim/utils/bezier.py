"""Utility functions related to Bézier curves."""

from __future__ import annotations

__all__ = [
    "bezier",
    "partial_bezier_points",
    "partial_quadratic_bezier_points",
    "split_bezier",
    "split_quadratic_bezier",
    "subdivide_bezier",
    "subdivide_quadratic_bezier",
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


import typing
from functools import reduce
from typing import Iterable

import numpy as np

from ..utils.simple_functions import choose
from ..utils.space_ops import cross2d, find_intersection


def bezier(
    points: np.ndarray,
) -> typing.Callable[[float], int | typing.Iterable]:
    """Classic implementation of a bezier curve.

    Parameters
    ----------
    points
        points defining the desired bezier curve.

    Returns
    -------
    typing.Callable[[float], typing.Union[int, typing.Iterable]]
        function describing the bezier curve.
    """
    P = np.asarray(points)
    n = P.shape[0] - 1

    if n == 0:

        def zero_bezier(t):
            return P[0]

        return zero_bezier

    if n == 1:

        def linear_bezier(t):
            return P[0] + t * (P[1] - P[0])

        return linear_bezier

    if n == 2:

        def quadratic_bezier(t):
            t2 = t * t
            mt = 1 - t
            mt2 = mt * mt
            return mt2 * P[0] + 2 * t * mt * P[1] + t2 * P[2]

        return quadratic_bezier

    if n == 3:

        def cubic_bezier(t):
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            return mt3 * P[0] + 3 * t * mt2 * P[1] + 3 * t2 * mt * P[2] + t3 * P[3]

        return cubic_bezier

    def nth_grade_bezier(t):
        B = P.copy()
        for i in range(n):
            # After the i-th iteration (i in [0, ..., n-1]) there are (n-i)
            # Bezier curves of grade (i+1) stored in the first n-i slots of B
            B[: n - i] += t * (B[1 : n - i + 1] - B[: n - i])
        # In the end, there shall be a single Bezier curve of grade n
        # stored in the first slot of B
        return B[0]

    return nth_grade_bezier


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

    """
    Original algorithm:

    a_to_1 = np.array([bezier(points[i:])(a) for i in range(n)])
    end_prop = (b - a) / (1.0 - a)
    return np.array([bezier(a_to_1[: i + 1])(end_prop) for i in range(n+1)])
    """


def partial_quadratic_bezier_points(points, a, b):
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

    arr = np.array(points, dtype=np.float64)
    # Border cases
    if a == 1:
        arr[:] = arr[-1]
        return arr
    if b == 0:
        arr[:] = arr[0]
        return arr

    # Current state: arr = [P0 P1 P2]
    if a != 0:
        arr[:-1] += a * (arr[1:] - arr[:-1])  # arr = [L0 L1 P2]
        arr[0] += a * (arr[1] - arr[0])  # arr = [Q0 L1 P2]

    # Current state: arr = [Q0 L1 P2] = [P0' P1' P2']
    if b != 1:
        if a != 0:
            mu = (1 - b) / (1 - a)
        else:
            mu = 1 - b
        arr[1:] += mu * (arr[:-1] - arr[1:])  # arr = [P0' L0' L1']
        arr[2] += mu * (arr[1] - arr[2])  # arr = [P0' L0' Q0']

    # TODO: this is converted to a list because the current implementation in
    # OpenGLVMobject.insert_n_curves_to_point_list does a list concatenation with +=.
    # Using an ndarray breaks many test cases. This should probably change.
    return list(arr)
    """
    Original algorithm:

    def curve(t):
        mt = 1 - t
        return points[0] * mt * mt + 2 * points[1] * t * mt + points[2] * t * t

    # bezier(points)
    h0 = curve(a) if a > 0 else points[0]
    h2 = curve(b) if b < 1 else points[2]
    h1_prime = (1 - a) * points[1] + a * points[2]
    end_prop = (b - a) / (1.0 - a)
    h1 = (1 - end_prop) * h0 + end_prop * h1_prime
    return [h0, h1, h2]
    """


def split_bezier(points: Iterable[float], t: float) -> np.ndarray:
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

    points = np.asarray(points)
    N, dim = points.shape
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

    return arr


def subdivide_bezier(points: Iterable[float], n_divisions: int) -> np.ndarray:
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
    points = np.asarray(points)
    if n_divisions == 1:
        return points

    N, dim = points.shape

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


def split_quadratic_bezier(points: np.ndarray, t: float) -> np.ndarray:
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
    s1 = interpolate(a1, h1, t)
    s2 = interpolate(h1, a2, t)
    p = interpolate(s1, s2, t)

    return np.array([a1, s1, p, p, s2, a2])


def subdivide_quadratic_bezier(points: Iterable[float], n: int) -> np.ndarray:
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
        tmp = split_quadratic_bezier(current, 1 / i)
        beziers[j] = tmp[:3]
        current = tmp[3:]
    return beziers.reshape(-1, 3)


def quadratic_bezier_remap(
    triplets: Iterable[Iterable[float]], new_number_of_curves: int
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
    new_triplets = np.empty((new_number_of_curves, 3, 3))
    idx = 0
    for triplet in triplets:
        if difference > 0:
            tmp_noc = int(np.ceil(difference / len(triplets))) + 1
            tmp = subdivide_quadratic_bezier(triplet, tmp_noc).reshape(-1, 3, 3)
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


# Linear interpolation variants
def interpolate(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start + alpha * (end - start)


def integer_interpolate(
    start: float,
    end: float,
    alpha: float,
) -> tuple[int, float]:
    """
    Alpha is a float between 0 and 1.  This returns
    an integer between start and end (inclusive) representing
    appropriate interpolation between them, along with a
    "residue" representing a new proportion between the
    returned integer and the next one of the
    list.

    For example, if start=0, end=10, alpha=0.46, This
    would return (4, 0.6).
    """
    if alpha >= 1:
        return (end - 1, 1.0)
    if alpha <= 0:
        return (start, 0)
    value = int(interpolate(start, end, alpha))
    residue = ((end - start) * alpha) % 1
    return (value, residue)


def mid(start: float, end: float) -> float:
    return (start + end) / 2.0


def inverse_interpolate(start: float, end: float, value: float) -> np.ndarray:
    return np.true_divide(value - start, end - start)


def match_interpolate(
    new_start: float,
    new_end: float,
    old_start: float,
    old_end: float,
    old_value: float,
) -> np.ndarray:
    return interpolate(
        new_start,
        new_end,
        inverse_interpolate(old_start, old_end, old_value),
    )


# Figuring out which Bezier curves most smoothly connect a sequence of points
def get_smooth_handle_points(
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return get_smooth_cubic_bezier_handle_points(anchors)


def get_smooth_cubic_bezier_handle_points(
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Given some anchors (points), compute handles so that the resulting Bezier curve is smooth.

    Parameters
    ----------
    anchors
        Anchors.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        Computed handles.
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
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global CP_CLOSED_MEMO
    global UP_CLOSED_MEMO
    """
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
    """

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
    anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    global CP_OPEN_MEMO
    """
    A system of equations must be solved to get the first handles of
    every Bèzier curve (referred to as H1).
    Then H2 (the second handles) can be obtained separately.
    The equations were obtained from:
    https://www.particleincell.com/2012/bezier-splines/
    http://www.jacos.nl/jacos_html/spline/theory/theory_2.html
    WARNING: the equations in the first webpage have some typos which
    were corrected in the comments.

    In general, if there are N+1 anchors there will be N Bezier curves
    and thus N pairs of handles to find. We must solve the following
    system of equations for the 1st handles (example for N = 5):

    [2 1 0 0 0]   [H1[0]]   [  A[0] + 2*A[1]]
    [1 4 1 0 0]   [H1[1]]   [4*A[1] + 2*A[2]]
    [0 1 4 1 0] @ [H1[2]] = [4*A[2] + 2*A[3]]
    [0 0 1 4 1]   [H1[3]]   [4*A[3] + 2*A[4]]
    [0 0 0 2 7]   [H1[4]]   [8*A[4] +   A[5]]

    which will be expressed as M @ H1 = d.
    M is a tridiagonal matrix, so the system can be solved in O(n) operations.
    Here we shall use Thomas' algorithm or the tridiagonal matrix algorithm,
    see https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

    Let a = [a[0], a[1], ..., a[N-2]] be the lower diagonal of N-1 elements,
    such that a[0]=a[1]=...=a[N-3] = 1, and A[N-2] = 2;
        b = [b[0], b[1], ..., b[N-2], b[N-1]] the main diagonal of N elements,
    such that b[0] = 2, b[1]=b[2]=...=b[N-2] = 4, and b[N-1] = 7;
    and c = [c[0], c[1], ..., c[N-2]] the upper diagonal of N-1 elements,
    such that c[0]=c[1]=...=c[N-2] = 1: this diagonal is filled with ones.

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

    Finally, we can do Backward Substitution to find H1:
    H1[N-1] = d'[N-1]
    H1[i]   = d'[i] - c'[i]*H1[i+1], for i in [N-2, ..., 0]

    Once we have H1, we can get H2 (the array of second handles) as follows:
    H2[i]   =   2*A[i+1]     - H1[i+1], for i in [0, ..., N-2]
    H2[N-1] = 0.5*A[N]   + 0.5*H1[N-1]

    As the matrix M always follows the same pattern, we can define a memo list
    for c' to avoid recalculation. We cannot do the same for d, however,
    because it is always a different vector.
    """

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
def diag_to_matrix(l_and_u: tuple[int, int], diag: np.ndarray) -> np.ndarray:
    """
    Converts array whose rows represent diagonal
    entries of a matrix into the matrix itself.
    See scipy.linalg.solve_banded
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


# Given 4 control points for a cubic bezier curve (or arrays of such)
# return control points for 2 quadratics (or 2n quadratics) approximating them.
def get_quadratic_approximation_of_cubic(a0, h0, h1, a1):
    a0 = np.array(a0, ndmin=2)
    h0 = np.array(h0, ndmin=2)
    h1 = np.array(h1, ndmin=2)
    a1 = np.array(a1, ndmin=2)
    # Tangent vectors at the start and end.
    T0 = h0 - a0
    T1 = a1 - h1

    # Search for inflection points.  If none are found, use the
    # midpoint as a cut point.
    # Based on http://www.caffeineowl.com/graphics/2d/vectorial/cubic-inflexion.html
    has_infl = np.ones(a0.shape[0], dtype=bool)

    p = h0 - a0
    q = h1 - 2 * h0 + a0
    r = a1 - 3 * h1 + 3 * h0 - a0

    a = cross2d(q, r)
    b = cross2d(p, r)
    c = cross2d(p, q)

    disc = b * b - 4 * a * c
    has_infl &= disc > 0
    sqrt_disc = np.sqrt(np.abs(disc))
    settings = np.seterr(all="ignore")
    ti_bounds = []
    for sgn in [-1, +1]:
        ti = (-b + sgn * sqrt_disc) / (2 * a)
        ti[a == 0] = (-c / b)[a == 0]
        ti[(a == 0) & (b == 0)] = 0
        ti_bounds.append(ti)
    ti_min, ti_max = ti_bounds
    np.seterr(**settings)
    ti_min_in_range = has_infl & (0 < ti_min) & (ti_min < 1)
    ti_max_in_range = has_infl & (0 < ti_max) & (ti_max < 1)

    # Choose a value of t which starts at 0.5,
    # but is updated to one of the inflection points
    # if they lie between 0 and 1

    t_mid = np.full(a0.shape[0], 0.5)
    t_mid[ti_min_in_range] = ti_min[ti_min_in_range]
    t_mid[ti_max_in_range] = ti_max[ti_max_in_range]

    m, n = a0.shape
    t_mid = t_mid.repeat(n).reshape((m, n))

    # Compute bezier point and tangent at the chosen value of t
    mid = bezier([a0, h0, h1, a1])(t_mid)
    Tm = bezier([h0 - a0, h1 - h0, a1 - h1])(t_mid)

    # Intersection between tangent lines at end points
    # and tangent in the middle
    i0 = find_intersection(a0, T0, mid, Tm)
    i1 = find_intersection(a1, T1, mid, Tm)

    m, n = np.shape(a0)
    result = np.empty((6 * m, n))
    result[0::6] = a0
    result[1::6] = i0
    result[2::6] = mid
    result[3::6] = mid
    result[4::6] = i1
    result[5::6] = a1
    return result


def is_closed(points: tuple[np.ndarray, np.ndarray]) -> bool:
    """Returns True if the curve given by the points is closed, by checking if its
    first and last points are close to each other.

    This function reimplements np.allclose (without a relative tolerance rtol),
    because repeated calling of np.allclose for only 2 points is inefficient.
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
    point: typing.Iterable[float | int],
    control_points: typing.Iterable[typing.Iterable[float | int]],
    round_to: float | int | None = 1e-6,
) -> np.ndarray:
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
        polynom_roots = bezier_polynom.roots()
        if len(polynom_roots) > 0:
            polynom_roots = np.around(polynom_roots, int(np.log10(1 / round_to)))
        roots.append(polynom_roots)

    roots = [[root for root in rootlist if root.imag == 0] for rootlist in roots]
    roots = reduce(np.intersect1d, roots)  # Get common roots.
    roots = np.array([r.real for r in roots if 0 <= r.real <= 1])
    return roots


def point_lies_on_bezier(
    point: typing.Iterable[float | int],
    control_points: typing.Iterable[typing.Iterable[float | int]],
    round_to: float | int | None = 1e-6,
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
