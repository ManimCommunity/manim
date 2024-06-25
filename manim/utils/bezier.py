"""Utility functions related to Bézier curves."""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "bezier",
    "partial_bezier_points",
    "partial_quadratic_bezier_points",
    "split_bezier",
    "subdivide_quadratic_bezier",
    "subdivide_bezier",
    "bezier_remap",
    "quadratic_bezier_remap",
    "interpolate",
    "integer_interpolate",
    "mid",
    "inverse_interpolate",
    "match_interpolate",
    "get_smooth_handle_points",
    "get_smooth_cubic_bezier_handle_points",
    "diag_to_matrix",
    "is_closed",
    "proportions_along_bezier_curve_for_point",
    "point_lies_on_bezier",
]


from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, overload

import numpy as np
from scipy import linalg

from manim.typing import PointDType

from ..utils.simple_functions import choose
from ..utils.space_ops import cross2d, find_intersection

if TYPE_CHECKING:
    import numpy.typing as npt

    from manim.typing import (
        BezierPoints,
        BezierPoints_Array,
        ColVector,
        MatrixMN,
        Point3D,
        Point3D_Array,
    )

# l is a commonly used name in linear algebra
# ruff: noqa: E741


def bezier(
    points: Sequence[Point3D] | Point3D_Array,
) -> Callable[[float], Point3D]:
    """Classic implementation of a bezier curve.

    Parameters
    ----------
    points
        points defining the desired bezier curve.

    Returns
    -------
        function describing the bezier curve.
        You can pass a t value between 0 and 1 to get the corresponding point on the curve.
    """
    n = len(points) - 1
    # Cubic Bezier curve
    if n == 3:
        return lambda t: np.asarray(
            (1 - t) ** 3 * points[0]
            + 3 * t * (1 - t) ** 2 * points[1]
            + 3 * (1 - t) * t**2 * points[2]
            + t**3 * points[3],
            dtype=PointDType,
        )
    # Quadratic Bezier curve
    if n == 2:
        return lambda t: np.asarray(
            (1 - t) ** 2 * points[0] + 2 * t * (1 - t) * points[1] + t**2 * points[2],
            dtype=PointDType,
        )

    return lambda t: np.asarray(
        np.asarray(
            [
                (((1 - t) ** (n - k)) * (t**k) * choose(n, k) * point)
                for k, point in enumerate(points)
            ],
            dtype=PointDType,
        ).sum(axis=0)
    )


def partial_bezier_points(points: BezierPoints, a: float, b: float) -> BezierPoints:
    r"""Given an array of ``points`` which define a Bézier curve, and two numbers :math:`a, b`
    such that :math:`0 \le a < b \le 1`, return an array of the same size, which describes the
    portion of the original Bézier curve on the interval :math:`[a, b]`.

    :func:`partial_bezier_points` is conceptually equivalent to calling :func:`split_bezier`
    twice and discarding unused Bézier curves, but this is more efficient and doesn't waste
    computations.

    .. seealso::
        See :func:`split_bezier` for an explanation on how to split Bézier curves.

    .. note::
        To find the portion of a Bézier curve with :math:`t` between :math:`a` and :math:`b`:

        1.  Split the curve at :math:`t = a` and extract its 2nd subcurve.
        2.  We cannot evaluate the new subcurve at :math:`t = b` because its range of values for :math:`t` is different.
            To find the correct value, we need to transform the interval :math:`[a, 1]` into :math:`[0, 1]`
            by first subtracting :math:`a` to get :math:`[0, 1-a]` and then dividing by :math:`1-a`. Thus, our new
            value must be :math:`t = \frac{b - a}{1 - a}`. Define :math:`u = \frac{b - a}{1 - a}`.
        3.  Split the subcurve at :math:`t = u` and extract its 1st subcurve.

        The final portion is a linear combination of points, and thus the process can be
        summarized as a linear transformation by some matrix in terms of :math:`a` and :math:`b`.
        This matrix is given explicitly for Bézier curves up to degree 3, which are often used in Manim.
        For higher degrees, the algorithm described previously is used.

        For the case of a quadratic Bézier curve:

        * Step 1:

        .. math::
            H'_1
            =
            \begin{pmatrix}
                (1-a)^2 & 2(1-a)a & a^2 \\
                0 & (1-a) & a \\
                0 & 0 & 1
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2
            \end{pmatrix}

        * Step 2:

        .. math::
            H''_0
            &=
            \begin{pmatrix}
                1 & 0 & 0 \\
                (1-u) & u & 0\\
                (1-u)^2 & 2(1-u)u & u^2
            \end{pmatrix}
            H'_1
            \\
            &
            \\
            &=
            \begin{pmatrix}
                1 & 0 & 0 \\
                (1-u) & u & 0\\
                (1-u)^2 & 2(1-u)u & u^2
            \end{pmatrix}
            \begin{pmatrix}
                (1-a)^2 & 2(1-a)a & a^2 \\
                0 & (1-a) & a \\
                0 & 0 & 1
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2
            \end{pmatrix}
            \\
            &
            \\
            &=
            \begin{pmatrix}
                (1-a)^2 & 2(1-a)a & a^2 \\
                (1-a)(1-b) & a(1-b) + (1-a)b & ab \\
                (1-b)^2 & 2(1-b)b & b^2
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2
            \end{pmatrix}

        from where one can define a :math:`(3, 3)` matrix :math:`P_2` which, when applied over
        the array of ``points``, will return the desired partial quadratic Bézier curve:

        .. math::
            P_2
            =
            \begin{pmatrix}
                (1-a)^2 & 2(1-a)a & a^2 \\
                (1-a)(1-b) & a(1-b) + (1-a)b & ab \\
                (1-b)^2 & 2(1-b)b & b^2
            \end{pmatrix}

        Similarly, for the cubic Bézier curve case, one can define the following
        :math:`(4, 4)` matrix :math:`P_3`:

        .. math::
            P_3
            =
            \begin{pmatrix}
                (1-a)^3 & 3(1-a)^2a & 3(1-a)a^2 & a^3 \\
                (1-a)^2(1-b) & 2(1-a)a(1-b) + (1-a)^2b & a^2(1-b) + 2(1-a)ab & a^2b \\
                (1-a)(1-b)^2 & a(1-b)^2 + 2(1-a)(1-b)b & 2a(1-b)b + (1-a)b^2 & ab^2 \\
                (1-b)^3 & 3(1-b)^2b & 3(1-b)b^2 & b^3
            \end{pmatrix}

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
    :class:`~.BezierPoints`
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

        portion_matrix = np.array(
            [
                [ma3, 3 * ma2 * a, 3 * ma * a2, a3],
                [ma2 * mb, 2 * ma * a * mb + ma2 * b, a2 * mb + 2 * ma * a * b, a2 * b],
                [ma * mb2, a * mb2 + 2 * ma * mb * b, 2 * a * mb * b + ma * b2, a * b2],
                [mb3, 3 * mb2 * b, 3 * mb * b2, b3],
            ]
        )
        return portion_matrix @ points

    if degree == 2:
        ma, mb = 1 - a, 1 - b

        portion_matrix = np.array(
            [
                [ma * ma, 2 * a * ma, a * a],
                [ma * mb, a * mb + ma * b, a * b],
                [mb * mb, 2 * b * mb, b * b],
            ]
        )
        return portion_matrix @ points

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
    arr = np.array(points, dtype=float)
    N = arr.shape[0]

    # Current state for an example Bézier curve C0 = [P0, P1, P2, P3]:
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


def partial_quadratic_bezier_points(points, a, b):
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


def split_bezier(points: BezierPoints, t: float) -> Point3D_Array:
    r"""Split a Bézier curve at argument ``t`` into two curves.

    .. note::

        .. seealso::
            `A Primer on Bézier Curves #10: Splitting curves. Pomax. <https://pomax.github.io/bezierinfo/#splitting>`_

        As an example for a cubic Bézier curve, let :math:`p_0, p_1, p_2, p_3` be the points
        needed for the curve :math:`C_0 = [p_0, \ p_1, \ p_2, \ p_3]`.

        Define the 3 linear Béziers :math:`L_0, L_1, L_2` as interpolations of :math:`p_0, p_1, p_2, p_3`:

        .. math::
            L_0(t) &= p_0 + t(p_1 - p_0) \\
            L_1(t) &= p_1 + t(p_2 - p_1) \\
            L_2(t) &= p_2 + t(p_3 - p_2)

        Define the 2 quadratic Béziers :math:`Q_0, Q_1` as interpolations of :math:`L_0, L_1, L_2`:

        .. math::
            Q_0(t) &= L_0(t) + t(L_1(t) - L_0(t)) \\
            Q_1(t) &= L_1(t) + t(L_2(t) - L_1(t))

        Then :math:`C_0` is the following interpolation of :math:`Q_0` and :math:`Q_1`:

        .. math::
            C_0(t) = Q_0(t) + t(Q_1(t) - Q_0(t))

        Evaluating :math:`C_0` at a value :math:`t=t'` splits :math:`C_0` into two cubic Béziers :math:`H_0`
        and :math:`H_1`, defined by some of the points we calculated earlier:

        .. math::
            H_0 &= [p_0, &\ L_0(t'), &\ Q_0(t'), &\ C_0(t') &] \\
            H_1 &= [p_0(t'), &\ Q_1(t'), &\ L_2(t'), &\ p_3 &]

        As the resulting curves are obtained from linear combinations of ``points``, everything can
        be encoded into a matrix for efficiency, which is done for Bézier curves of degree up to 3.

        .. seealso::
            `A Primer on Bézier Curves #11: Splitting curves using matrices. Pomax. <https://pomax.github.io/bezierinfo/#matrixsplit>`_

        For the simpler case of a quadratic Bézier curve:

        .. math::
            H_0
            &=
            \begin{pmatrix}
                p_0 \\
                (1-t) p_0 + t p_1 \\
                (1-t)^2 p_0 + 2(1-t)t p_1 + t^2 p_2 \\
            \end{pmatrix}
            &=
            \begin{pmatrix}
                1 & 0 & 0 \\
                (1-t) & t & 0\\
                (1-t)^2 & 2(1-t)t & t^2
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2
            \end{pmatrix}
            \\
            &
            \\
            H_1
            &=
            \begin{pmatrix}
                (1-t)^2 p_0 + 2(1-t)t p_1 + t^2 p_2 \\
                (1-t) p_1 + t p_2 \\
                p_2
            \end{pmatrix}
            &=
            \begin{pmatrix}
                (1-t)^2 & 2(1-t)t & t^2 \\
                0 & (1-t) & t \\
                0 & 0 & 1
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2
            \end{pmatrix}

        from where one can define a :math:`(6, 3)` split matrix :math:`S_2` which can multiply
        the array of ``points`` to compute the return value:

        .. math::
            S_2
            &=
            \begin{pmatrix}
                1 & 0 & 0 \\
                (1-t) & t & 0 \\
                (1-t)^2 & 2(1-t)t & t^2 \\
                (1-t)^2 & 2(1-t)t & t^2 \\
                0 & (1-t) & t \\
                0 & 0 & 1
            \end{pmatrix}
            \\
            &
            \\
            S_2 P
            &=
            \begin{pmatrix}
                1 & 0 & 0 \\
                (1-t) & t & 0 \\
                (1-t)^2 & 2(1-t)t & t^2 \\
                (1-t)^2 & 2(1-t)t & t^2 \\
                0 & (1-t) & t \\
                0 & 0 & 1
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2
            \end{pmatrix}
            =
            \begin{pmatrix}
                \vert \\
                H_0 \\
                \vert \\
                \vert \\
                H_1 \\
                \vert
            \end{pmatrix}

        For the previous example with a cubic Bézier curve:

        .. math::
            H_0
            &=
            \begin{pmatrix}
                p_0 \\
                (1-t) p_0 + t p_1 \\
                (1-t)^2 p_0 + 2(1-t)t p_1 + t^2 p_2 \\
                (1-t)^3 p_0 + 3(1-t)^2 t p_1 + 3(1-t)t^2 p_2 + t^3 p_3
            \end{pmatrix}
            &=
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                (1-t) & t & 0 & 0 \\
                (1-t)^2 & 2(1-t)t & t^2 & 0 \\
                (1-t)^3 & 3(1-t)^2 t & 3(1-t)t^2 & t^3
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2 \\
                p_3
            \end{pmatrix}
            \\
            &
            \\
            H_1
            &=
            \begin{pmatrix}
                (1-t)^3 p_0 + 3(1-t)^2 t p_1 + 3(1-t)t^2 p_2 + t^3 p_3 \\
                (1-t)^2 p_1 + 2(1-t)t p_2 + t^2 p_3 \\
                (1-t) p_2 + t p_3 \\
                p_3
            \end{pmatrix}
            &=
            \begin{pmatrix}
                (1-t)^3 & 3(1-t)^2 t & 3(1-t)t^2 & t^3 \\
                0 & (1-t)^2 & 2(1-t)t & t^2 \\
                0 & 0 & (1-t) & t \\
                0 & 0 & 0 & 1
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2 \\
                p_3
            \end{pmatrix}

        from where one can define a :math:`(8, 4)` split matrix :math:`S_3` which can multiply
        the array of ``points`` to compute the return value:

        .. math::
            S_3
            &=
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                (1-t) & t & 0 & 0 \\
                (1-t)^2 & 2(1-t)t & t^2 & 0 \\
                (1-t)^3 & 3(1-t)^2 t & 3(1-t)t^2 & t^3 \\
                (1-t)^3 & 3(1-t)^2 t & 3(1-t)t^2 & t^3 \\
                0 & (1-t)^2 & 2(1-t)t & t^2 \\
                0 & 0 & (1-t) & t \\
                0 & 0 & 0 & 1
            \end{pmatrix}
            \\
            &
            \\
            S_3 P
            &=
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                (1-t) & t & 0 & 0 \\
                (1-t)^2 & 2(1-t)t & t^2 & 0 \\
                (1-t)^3 & 3(1-t)^2 t & 3(1-t)t^2 & t^3 \\
                (1-t)^3 & 3(1-t)^2 t & 3(1-t)t^2 & t^3 \\
                0 & (1-t)^2 & 2(1-t)t & t^2 \\
                0 & 0 & (1-t) & t \\
                0 & 0 & 0 & 1
            \end{pmatrix}
            \begin{pmatrix}
                p_0 \\
                p_1 \\
                p_2 \\
                p_3
            \end{pmatrix}
            =
            \begin{pmatrix}
                \vert \\
                H_0 \\
                \vert \\
                \vert \\
                H_1 \\
                \vert
            \end{pmatrix}

    Parameters
    ----------
    points
        The control points of the Bézier curve.

    t
        The ``t``-value at which to split the Bézier curve.

    Returns
    -------
    :class:`~.Point3D_Array`
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

        # Split matrix S3 explained in the docstring
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

        # Split matrix S2 explained in the docstring
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

    # Example for a cubic Bézier
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


# Memos explained in subdivide_bezier docstring
SUBDIVISION_MATRICES = [{} for i in range(4)]


def _get_subdivision_matrix(n_points: int, n_divisions: int) -> MatrixMN:
    """Gets the matrix which subdivides a Bézier curve of
    ``n_points`` control points into ``n_divisions`` parts.

    Auxiliary function for :func:`subdivide_bezier`. See its
    docstrings for an explanation of the matrix build process.

    Parameters
    ----------
    n_points
        The number of control points of the Bézier curve to
        subdivide. This function only handles up to 4 points.
    n_divisions
        The number of parts to subdivide the Bézier curve into.

    Returns
    -------
    MatrixMN
        The matrix which, upon multiplying the control points of the
        Bézier curve, subdivides it into ``n_divisions`` parts.
    """
    if n_points not in (1, 2, 3, 4):
        raise NotImplementedError(
            "This function does not support subdividing Bézier "
            "curves with 0 or more than 4 control points."
        )

    subdivision_matrix = SUBDIVISION_MATRICES[n_points - 1].get(n_divisions, None)
    if subdivision_matrix is not None:
        return subdivision_matrix

    subdivision_matrix = np.empty((n_points * n_divisions, n_points))

    # Cubic Bézier
    if n_points == 4:
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
            nmim12 = nmim1 * nmim1
            nmim13 = nmim12 * nmim1

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

    # Quadratic Bézier
    elif n_points == 3:
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

    # Linear Bézier (straight line)
    elif n_points == 2:
        aux_range = np.arange(n_divisions + 1)
        subdivision_matrix[::2, 1] = aux_range[:-1]
        subdivision_matrix[1::2, 1] = aux_range[1:]
        subdivision_matrix[:, 0] = subdivision_matrix[::-1, 1]
        subdivision_matrix /= n_divisions

    # Zero-degree Bézier (single point)
    elif n_points == 1:
        subdivision_matrix[:] = 1

    SUBDIVISION_MATRICES[n_points - 1][n_divisions] = subdivision_matrix
    return subdivision_matrix


def subdivide_bezier(points: BezierPoints, n_divisions: int) -> Point3D_Array:
    r"""Subdivide a Bézier curve into :math:`n` subcurves which have the same shape.

    The points at which the curve is split are located at the
    arguments :math:`t = \frac{i}{n}`, for :math:`i \in \{1, ..., n-1\}`.

    .. seealso::

        * See :func:`split_bezier` for an explanation on how to split Bézier curves.
        * See :func:`partial_bezier_points` for an extra understanding of this function.


    .. note::
        The resulting subcurves can be expressed as linear combinations of
        ``points``, which can be encoded in a single matrix that is precalculated
        for 2nd and 3rd degree Bézier curves.

        As an example for a quadratic Bézier curve: taking inspiration from the
        explanation in :func:`partial_bezier_points`, where the following matrix
        :math:`P_2` was defined to extract the portion of a quadratic Bézier
        curve for :math:`t \in [a, b]`:

        .. math::
            P_2
            =
            \begin{pmatrix}
                (1-a)^2 & 2(1-a)a & a^2 \\
                (1-a)(1-b) & a(1-b) + (1-a)b & ab \\
                (1-b)^2 & 2(1-b)b & b^2
            \end{pmatrix}

        the plan is to replace :math:`[a, b]` with
        :math:`\left[ \frac{i-1}{n}, \frac{i}{n} \right], \ \forall i \in \{1, ..., n\}`.

        As an example for :math:`n = 2` divisions, construct :math:`P_1` for
        the interval :math:`\left[ 0, \frac{1}{2} \right]`, and :math:`P_2` for the
        interval :math:`\left[ \frac{1}{2}, 1 \right]`:

        .. math::
            P_1
            =
            \begin{pmatrix}
                1 & 0 & 0 \\
                0.5 & 0.5 & 0 \\
                0.25 & 0.5 & 0.25
            \end{pmatrix}
            ,
            \quad
            P_2
            =
            \begin{pmatrix}
                0.25 & 0.5 & 0.25 \\
                0 & 0.5 & 0.5 \\
                0 & 0 & 1
            \end{pmatrix}

        Therefore, the following :math:`(6, 3)` subdivision matrix :math:`D_2` can be
        constructed, which will subdivide an array of ``points`` into 2 parts:

        .. math::
            D_2
            =
            \begin{pmatrix}
                M_1 \\
                M_2
            \end{pmatrix}
            =
            \begin{pmatrix}
                1 & 0 & 0 \\
                0.5 & 0.5 & 0 \\
                0.25 & 0.5 & 0.25 \\
                0.25 & 0.5 & 0.25 \\
                0 & 0.5 & 0.5 \\
                0 & 0 & 1
            \end{pmatrix}

        For quadratic and cubic Bézier curves, the subdivision matrices are memoized for
        efficiency. For higher degree curves, an iterative algorithm inspired by the
        one from :func:`split_bezier` is used instead.

    .. image:: /_static/bezier_subdivision_example.png

    Parameters
    ----------
    points
        The control points of the Bézier curve.

    n_divisions
        The number of curves to subdivide the Bézier curve into

    Returns
    -------
    :class:`~.Point3D_Array`
        An array containing the points defining the new :math:`n` subcurves.
    """
    if n_divisions == 1:
        return points

    points = np.asarray(points)
    N, dim = points.shape

    if N <= 4:
        subdivision_matrix = _get_subdivision_matrix(N, n_divisions)
        return subdivision_matrix @ points

    # Fallback case for an nth degree Bézier: successive splitting
    beziers = np.empty((n_divisions, N, dim))
    beziers[-1] = points
    for curve_num in range(n_divisions - 1, 0, -1):
        curr = beziers[curve_num]
        prev = beziers[curve_num - 1]
        prev[0] = curr[0]
        a = (n_divisions - curve_num) / (n_divisions - curve_num + 1)
        # Current state for an example cubic Bézier curve:
        # prev = [P0 .. .. ..]
        # curr = [P0 P1 P2 P3]
        for i in range(1, N):
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
        An array of multiple Bézier curves of degree :math:`d` to be remapped. The shape of this array
        must be ``(current_number_of_curves, nppc, dim)``, where:

            *   ``current_number_of_curves`` is the current amount of curves in the array ``bezier_tuples``,
            *   ``nppc`` is the amount of points per curve, such that their degree is ``nppc-1``, and
            *   ``dim`` is the dimension of the points, usually :math:`3`.
    new_number_of_curves
        The number of curves that the output will contain. This needs to be higher than the current number.

    Returns
    -------
    :class:`~.BezierPoints_Array`
        The new array of shape ``(new_number_of_curves, nppc, dim)``,
        containing the new Bézier curves after the remap.
    """
    bezier_tuples = np.asarray(bezier_tuples)
    current_number_of_curves, nppc, dim = bezier_tuples.shape
    # This is an array with values ranging from 0
    # up to curr_num_curves,  with repeats such that
    # its total length is target_num_curves.  For example,
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
    new_triplets = np.zeros((new_number_of_curves, 3, 3))
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


@overload
def interpolate(start: float, end: float, alpha: float) -> float: ...


@overload
def interpolate(start: Point3D, end: Point3D, alpha: float) -> Point3D: ...


def interpolate(
    start: int | float | Point3D, end: int | float | Point3D, alpha: float | Point3D
) -> float | Point3D:
    return (1 - alpha) * start + alpha * end


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
def mid(start: float, end: float) -> float: ...


@overload
def mid(start: Point3D, end: Point3D) -> Point3D: ...


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
def inverse_interpolate(start: float, end: float, value: float) -> float: ...


@overload
def inverse_interpolate(start: float, end: float, value: Point3D) -> Point3D: ...


@overload
def inverse_interpolate(start: Point3D, end: Point3D, value: Point3D) -> Point3D: ...


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
) -> float: ...


@overload
def match_interpolate(
    new_start: float,
    new_end: float,
    old_start: float,
    old_end: float,
    old_value: Point3D,
) -> Point3D: ...


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


def get_smooth_quadratic_bezier_handle_points(points: FloatArray) -> FloatArray:
    """
    Figuring out which bezier curves most smoothly connect a sequence of points.

    Given three successive points, P0, P1 and P2, you can compute that by defining
    h = (1/4) P0 + P1 - (1/4)P2, the bezier curve defined by (P0, h, P1) will pass
    through the point P2.

    So for a given set of four successive points, P0, P1, P2, P3, if we want to add
    a handle point h between P1 and P2 so that the quadratic bezier (P1, h, P2) is
    part of a smooth curve passing through all four points, we calculate one solution
    for h that would produce a parbola passing through P3, call it smooth_to_right, and
    another that would produce a parabola passing through P0, call it smooth_to_left,
    and use the midpoint between the two.
    """
    if len(points) == 2:
        return midpoint(*points)
    smooth_to_right, smooth_to_left = (
        0.25 * ps[0:-2] + ps[1:-1] - 0.25 * ps[2:] for ps in (points, points[::-1])
    )
    if np.isclose(points[0], points[-1]).all():
        last_str = 0.25 * points[-2] + points[-1] - 0.25 * points[1]
        last_stl = 0.25 * points[1] + points[0] - 0.25 * points[-2]
    else:
        last_str = smooth_to_left[0]
        last_stl = smooth_to_right[0]
    handles = 0.5 * np.vstack([smooth_to_right, [last_str]])
    handles += 0.5 * np.vstack([last_stl, smooth_to_left[::-1]])
    return handles


def get_smooth_cubic_bezier_handle_points(
    points: Point3D_Array,
) -> tuple[BezierPoints, BezierPoints]:
    points = np.asarray(points)
    num_handles = len(points) - 1
    dim = points.shape[1]
    if num_handles < 1:
        return np.zeros((0, dim)), np.zeros((0, dim))
    # Must solve 2*num_handles equations to get the handles.
    # l and u are the number of lower an upper diagonal rows
    # in the matrix to solve.
    l, u = 2, 1
    # diag is a representation of the matrix in diagonal form
    # See https://www.particleincell.com/2012/bezier-splines/
    # for how to arrive at these equations
    diag: MatrixMN = np.zeros((l + u + 1, 2 * num_handles))
    diag[0, 1::2] = -1
    diag[0, 2::2] = 1
    diag[1, 0::2] = 2
    diag[1, 1::2] = 1
    diag[2, 1:-2:2] = -2
    diag[3, 0:-3:2] = 1
    # last
    diag[2, -2] = -1
    diag[1, -1] = 2
    # This is the b as in Ax = b, where we are solving for x,
    # and A is represented using diag.  However, think of entries
    # to x and b as being points in space, not numbers
    b: Point3D_Array = np.zeros((2 * num_handles, dim))
    b[1::2] = 2 * points[1:]
    b[0] = points[0]
    b[-1] = points[-1]

    def solve_func(b: ColVector) -> ColVector | MatrixMN:
        return linalg.solve_banded((l, u), diag, b)  # type: ignore

    use_closed_solve_function = is_closed(points)
    if use_closed_solve_function:
        # Get equations to relate first and last points
        matrix = diag_to_matrix((l, u), diag)
        # last row handles second derivative
        matrix[-1, [0, 1, -2, -1]] = [2, -1, 1, -2]
        # first row handles first derivative
        matrix[0, :] = np.zeros(matrix.shape[1])
        matrix[0, [0, -1]] = [1, 1]
        b[0] = 2 * points[0]
        b[-1] = np.zeros(dim)

        def closed_curve_solve_func(b: ColVector) -> ColVector | MatrixMN:
            return linalg.solve(matrix, b)  # type: ignore

    handle_pairs = np.zeros((2 * num_handles, dim))
    for i in range(dim):
        if use_closed_solve_function:
            handle_pairs[:, i] = closed_curve_solve_func(b[:, i])
        else:
            handle_pairs[:, i] = solve_func(b[:, i])
    return handle_pairs[0::2], handle_pairs[1::2]


def get_smooth_handle_points(
    points: BezierPoints,
) -> tuple[BezierPoints, BezierPoints]:
    """Given some anchors (points), compute handles so the resulting bezier curve is smooth.

    Parameters
    ----------
    points
        Anchors.

    Returns
    -------
    typing.Tuple[np.ndarray, np.ndarray]
        Computed handles.
    """
    # NOTE points here are anchors.
    points = np.asarray(points)
    num_handles = len(points) - 1
    dim = points.shape[1]
    if num_handles < 1:
        return np.zeros((0, dim)), np.zeros((0, dim))
    # Must solve 2*num_handles equations to get the handles.
    # l and u are the number of lower an upper diagonal rows
    # in the matrix to solve.
    l, u = 2, 1
    # diag is a representation of the matrix in diagonal form
    # See https://www.particleincell.com/2012/bezier-splines/
    # for how to arrive at these equations
    diag: MatrixMN = np.zeros((l + u + 1, 2 * num_handles))
    diag[0, 1::2] = -1
    diag[0, 2::2] = 1
    diag[1, 0::2] = 2
    diag[1, 1::2] = 1
    diag[2, 1:-2:2] = -2
    diag[3, 0:-3:2] = 1
    # last
    diag[2, -2] = -1
    diag[1, -1] = 2
    # This is the b as in Ax = b, where we are solving for x,
    # and A is represented using diag.  However, think of entries
    # to x and b as being points in space, not numbers
    b = np.zeros((2 * num_handles, dim))
    b[1::2] = 2 * points[1:]
    b[0] = points[0]
    b[-1] = points[-1]

    def solve_func(b: ColVector) -> ColVector | MatrixMN:
        return linalg.solve_banded((l, u), diag, b)  # type: ignore

    use_closed_solve_function = is_closed(points)
    if use_closed_solve_function:
        # Get equations to relate first and last points
        matrix = diag_to_matrix((l, u), diag)
        # last row handles second derivative
        matrix[-1, [0, 1, -2, -1]] = [2, -1, 1, -2]
        # first row handles first derivative
        matrix[0, :] = np.zeros(matrix.shape[1])
        matrix[0, [0, -1]] = [1, 1]
        b[0] = 2 * points[0]
        b[-1] = np.zeros(dim)

        def closed_curve_solve_func(b: ColVector) -> ColVector | MatrixMN:
            return linalg.solve(matrix, b)  # type: ignore

    handle_pairs = np.zeros((2 * num_handles, dim))
    for i in range(dim):
        if use_closed_solve_function:
            handle_pairs[:, i] = closed_curve_solve_func(b[:, i])
        else:
            handle_pairs[:, i] = solve_func(b[:, i])
    return handle_pairs[0::2], handle_pairs[1::2]


def diag_to_matrix(
    l_and_u: tuple[int, int], diag: npt.NDArray[Any]
) -> npt.NDArray[Any]:
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
def get_quadratic_approximation_of_cubic(
    a0: Point3D, h0: Point3D, h1: Point3D, a1: Point3D
) -> BezierPoints:
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
    has_infl = np.ones(len(a0), dtype=bool)

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

    t_mid = 0.5 * np.ones(len(a0))
    t_mid[ti_min_in_range] = ti_min[ti_min_in_range]
    t_mid[ti_max_in_range] = ti_max[ti_max_in_range]

    m, n = a0.shape
    t_mid = t_mid.repeat(n).reshape((m, n))

    # Compute bezier point and tangent at the chosen value of t (these are vectorized)
    mid = bezier([a0, h0, h1, a1])(t_mid)  # type: ignore
    Tm = bezier([h0 - a0, h1 - h0, a1 - h1])(t_mid)  # type: ignore

    # Intersection between tangent lines at end points
    # and tangent in the middle
    i0 = find_intersection(a0, T0, mid, Tm)
    i1 = find_intersection(a1, T1, mid, Tm)

    m, n = np.shape(a0)
    result = np.zeros((6 * m, n))
    result[0::6] = a0
    result[1::6] = i0
    result[2::6] = mid
    result[3::6] = mid
    result[4::6] = i1
    result[5::6] = a1
    return result


def is_closed(points: Point3D_Array) -> bool:
    """Returns ``True`` if the spline given by ``points`` is closed, by
    checking if its first and last points are close to each other, or``False``
    otherwise.

    .. note::

        This function reimplements :meth:`np.allclose`, because repeated
        calling of :meth:`np.allclose` for only 2 points is inefficient.

    Parameters
    ----------
    points
        An array of points defining a spline.

    Returns
    -------
    :class:`bool`
        Whether the first and last points of the array are close enough or not
        to be considered the same, thus considering the defined spline as
        closed.

    Examples
    --------
    .. code-block:: pycon

        >>> import numpy as np
        >>> from manim import is_closed
        >>> is_closed(
        ...     np.array(
        ...         [
        ...             [0, 0, 0],
        ...             [1, 2, 3],
        ...             [3, 2, 1],
        ...             [0, 0, 0],
        ...         ]
        ...     )
        ... )
        True
        >>> is_closed(
        ...     np.array(
        ...         [
        ...             [0, 0, 0],
        ...             [1, 2, 3],
        ...             [3, 2, 1],
        ...             [1e-10, 1e-10, 1e-10],
        ...         ]
        ...     )
        ... )
        True
        >>> is_closed(
        ...     np.array(
        ...         [
        ...             [0, 0, 0],
        ...             [1, 2, 3],
        ...             [3, 2, 1],
        ...             [1e-2, 1e-2, 1e-2],
        ...         ]
        ...     )
        ... )
        False
    """
    start, end = points[0], points[-1]
    rtol = 1e-5
    atol = 1e-8
    tolerance = atol + rtol * start
    if abs(end[0] - start[0]) > tolerance[0]:
        return False
    if abs(end[1] - start[1]) > tolerance[1]:
        return False
    if abs(end[2] - start[2]) > tolerance[2]:
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
