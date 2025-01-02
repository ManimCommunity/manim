"""Utility functions related to Bézier curves."""

from __future__ import annotations

__all__ = [
    "bezier",
    "partial_bezier_points",
    "split_bezier",
    "subdivide_bezier",
    "bezier_remap",
    "interpolate",
    "integer_interpolate",
    "mid",
    "inverse_interpolate",
    "match_interpolate",
    "get_smooth_cubic_bezier_handle_points",
    "is_closed",
    "proportions_along_bezier_curve_for_point",
    "point_lies_on_bezier",
]


from collections.abc import Sequence
from functools import reduce
from typing import TYPE_CHECKING, Callable, overload

import numpy as np

from manim.utils.simple_functions import choose

if TYPE_CHECKING:
    from manim.typing import (
        BezierPoints,
        BezierPoints_Array,
        BezierPointsLike,
        BezierPointsLike_Array,
        ColVector,
        MatrixMN,
        Point3D,
        Point3D_Array,
        Point3DLike,
        Point3DLike_Array,
        QuadraticBezierPath,
        QuadraticSpline,
        Spline,
    )

# l is a commonly used name in linear algebra
# ruff: noqa: E741


@overload
def bezier(
    points: BezierPointsLike,
) -> Callable[[float | ColVector], Point3D | Point3D_Array]: ...


@overload
def bezier(
    points: Sequence[Point3DLike_Array],
) -> Callable[[float | ColVector], Point3D_Array]: ...


def bezier(
    points: Point3D_Array | Sequence[Point3D_Array],
) -> Callable[[float | ColVector], Point3D_Array]:
    """Classic implementation of a Bézier curve.

    Parameters
    ----------
    points
        :math:`(d+1, 3)`-shaped array of :math:`d+1` control points defining a single Bézier
        curve of degree :math:`d`. Alternatively, for vectorization purposes, ``points`` can
        also be a :math:`(d+1, M, 3)`-shaped sequence of :math:`d+1` arrays of :math:`M`
        control points each, which define `M` Bézier curves instead.

    Returns
    -------
    bezier_func : :class:`typing.Callable` [[:class:`float` | :class:`~.ColVector`], :class:`~.Point3D` | :class:`~.Point3D_Array`]
        Function describing the Bézier curve. The behaviour of this function depends on
        the shape of ``points``:

            *   If ``points`` was a :math:`(d+1, 3)` array representing a single Bézier curve,
                then ``bezier_func`` can receive either:

                *   a :class:`float` ``t``, in which case it returns a
                    single :math:`(1, 3)`-shaped :class:`~.Point3D` representing the evaluation
                    of the Bézier at ``t``, or
                *   an :math:`(n, 1)`-shaped :class:`~.ColVector`
                    containing :math:`n` values to evaluate the Bézier curve at, returning instead
                    an :math:`(n, 3)`-shaped :class:`~.Point3D_Array` containing the points
                    resulting from evaluating the Bézier at each of the :math:`n` values.
                .. warning::
                    If passing a vector of :math:`t`-values to ``bezier_func``, it **must**
                    be a column vector/matrix of shape :math:`(n, 1)`. Passing an 1D array of
                    shape :math:`(n,)` is not supported and **will result in undefined behaviour**.

            *   If ``points`` was a :math:`(d+1, M, 3)` array describing :math:`M` Bézier curves,
                then ``bezier_func`` can receive either:

                *   a :class:`float` ``t``, in which case it returns an
                    :math:`(M, 3)`-shaped :class:`~.Point3D_Array` representing the evaluation
                    of the :math:`M` Bézier curves at the same value ``t``, or
                *   an :math:`(M, 1)`-shaped
                    :class:`~.ColVector` containing :math:`M` values, such that the :math:`i`-th
                    Bézier curve defined by ``points`` is evaluated at the corresponding :math:`i`-th
                    value in ``t``, returning again an :math:`(M, 3)`-shaped :class:`~.Point3D_Array`
                    containing those :math:`M` evaluations.
                .. warning::
                    Unlike the previous case, if you pass a :class:`~.ColVector` to ``bezier_func``,
                    it **must** contain exactly :math:`M` values, each value for each of the :math:`M`
                    Bézier curves defined by ``points``. Any array of shape other than :math:`(M, 1)`
                    **will result in undefined behaviour**.
    """
    P = np.asarray(points)
    degree = P.shape[0] - 1

    if degree == 0:

        def zero_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            return np.ones_like(t) * P[0]

        return zero_bezier

    if degree == 1:

        def linear_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            return P[0] + t * (P[1] - P[0])

        return linear_bezier

    if degree == 2:

        def quadratic_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            t2 = t * t
            mt = 1 - t
            mt2 = mt * mt
            return mt2 * P[0] + 2 * t * mt * P[1] + t2 * P[2]

        return quadratic_bezier

    if degree == 3:

        def cubic_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
            t2 = t * t
            t3 = t2 * t
            mt = 1 - t
            mt2 = mt * mt
            mt3 = mt2 * mt
            return mt3 * P[0] + 3 * t * mt2 * P[1] + 3 * t2 * mt * P[2] + t3 * P[3]

        return cubic_bezier

    def nth_grade_bezier(t: float | ColVector) -> Point3D | Point3D_Array:
        is_scalar = not isinstance(t, np.ndarray)
        if is_scalar:
            B = np.empty((1, *P.shape))
        else:
            assert isinstance(t, np.ndarray)
            t = t.reshape(-1, *[1 for dim in P.shape])
            B = np.empty((t.shape[0], *P.shape))
        B[:] = P

        for i in range(degree):
            # After the i-th iteration (i in [0, ..., d-1]) there are evaluations at t
            # of (d-i) Bezier curves of grade (i+1), stored in the first d-i slots of B
            B[:, : degree - i] += t * (B[:, 1 : degree - i + 1] - B[:, : degree - i])

        # In the end, there shall be the evaluation at t of a single Bezier curve of
        # grade d, stored in the first slot of B
        if is_scalar:
            val: Point3D = B[0, 0]
            return val
        return B[:, 0]

    return nth_grade_bezier


def partial_bezier_points(points: BezierPointsLike, a: float, b: float) -> BezierPoints:
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


def split_bezier(points: BezierPointsLike, t: float) -> Spline:
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


# Memos explained in subdivide_bezier docstring
SUBDIVISION_MATRICES: list[dict[int, MatrixMN]] = [{} for i in range(4)]


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


def subdivide_bezier(points: BezierPointsLike, n_divisions: int) -> Spline:
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
    :class:`~.Spline`
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
    bezier_tuples: BezierPointsLike_Array,
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


# Linear interpolation variants


@overload
def interpolate(start: float, end: float, alpha: float) -> float: ...


@overload
def interpolate(start: float, end: float, alpha: ColVector) -> ColVector: ...


@overload
def interpolate(start: Point3D, end: Point3D, alpha: float) -> Point3D: ...


@overload
def interpolate(start: Point3D, end: Point3D, alpha: ColVector) -> Point3D_Array: ...


def interpolate(
    start: float | Point3D,
    end: float | Point3D,
    alpha: float | ColVector,
) -> float | ColVector | Point3D | Point3D_Array:
    """Linearly interpolates between two values ``start`` and ``end``.

    Parameters
    ----------
    start
        The start of the range.
    end
        The end of the range.
    alpha
        A float between 0 and 1, or an :math:`(n, 1)` column vector containing
        :math:`n` floats between 0 and 1 to interpolate in a vectorized fashion.

    Returns
    -------
    :class:`float` | :class:`~.ColVector` | :class:`~.Point3D` | :class:`~.Point3D_Array`
        The result of the linear interpolation.

        *   If ``start`` and ``end`` are of type :class:`float`, and:

            * ``alpha`` is also a :class:`float`, the return is simply another :class:`float`.
            * ``alpha`` is a :class:`~.ColVector`, the return is another :class:`~.ColVector`.

        *   If ``start`` and ``end`` are of type :class:`~.Point3D`, and:

            * ``alpha`` is a :class:`float`, the return is another :class:`~.Point3D`.
            * ``alpha`` is a :class:`~.ColVector`, the return is a :class:`~.Point3D_Array`.
    """
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
    start: float | Point3D,
    end: float | Point3D,
    value: float | Point3D,
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
        np.float64(0.5)

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
    np.float64(50.0)
    """
    old_alpha = inverse_interpolate(old_start, old_end, old_value)
    return interpolate(
        new_start,
        new_end,
        old_alpha,
    )


# Figuring out which Bézier curves most smoothly connect a sequence of points
def get_smooth_cubic_bezier_handle_points(
    anchors: Point3DLike_Array,
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
    :class:`tuple` [:class:`~.Point3D_Array`, :class:`~.Point3D_Array`]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the cubic spline, and the other containing the 2nd handles.
    """
    anchors = np.asarray(anchors)
    n_anchors = anchors.shape[0]

    # If there's a single anchor, there's no Bézier curve.
    # Return empty arrays.
    if n_anchors == 1:
        dim = anchors.shape[1]
        return np.zeros((0, dim)), np.zeros((0, dim))

    # If there are only two anchors (thus only one pair of handles),
    # they can only be an interpolation of these two anchors with alphas
    # 1/3 and 2/3, which will draw a straight line between the anchors.
    if n_anchors == 2:
        val = interpolate(anchors[0], anchors[1], np.array([[1 / 3], [2 / 3]]))
        return (val[0], val[1])

    # Handle different cases depending on whether the points form a closed
    # curve or not
    curve_is_closed = is_closed(anchors)
    if curve_is_closed:
        return get_smooth_closed_cubic_bezier_handle_points(anchors)
    else:
        return get_smooth_open_cubic_bezier_handle_points(anchors)


CP_CLOSED_MEMO = np.array([1 / 3])
UP_CLOSED_MEMO = np.array([1 / 3])


def get_smooth_closed_cubic_bezier_handle_points(
    anchors: Point3DLike_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    r"""Special case of :func:`get_smooth_cubic_bezier_handle_points`,
    when the ``anchors`` form a closed loop.

    .. note::
        A system of equations must be solved to get the first handles of
        every Bézier curve (referred to as :math:`H_1`).
        Then :math:`H_2` (the second handles) can be obtained separately.

        .. seealso::
            The equations were obtained from:

            * `Conditions on control points for continuous curvature. (2016). Jaco Stuifbergen. <http://www.jacos.nl/jacos_html/spline/theory/theory_2.html>`_

        In general, if there are :math:`N+1` anchors, there will be :math:`N` Bézier curves
        and thus :math:`N` pairs of handles to find. We must solve the following
        system of equations for the 1st handles (example for :math:`N = 5`):

        .. math::
            \begin{pmatrix}
                4 & 1 & 0 & 0 & 1 \\
                1 & 4 & 1 & 0 & 0 \\
                0 & 1 & 4 & 1 & 0 \\
                0 & 0 & 1 & 4 & 1 \\
                1 & 0 & 0 & 1 & 4
            \end{pmatrix}
            \begin{pmatrix}
                H_{1,0} \\
                H_{1,1} \\
                H_{1,2} \\
                H_{1,3} \\
                H_{1,4}
            \end{pmatrix}
            =
            \begin{pmatrix}
                4A_0 + 2A_1 \\
                4A_1 + 2A_2 \\
                4A_2 + 2A_3 \\
                4A_3 + 2A_4 \\
                4A_4 + 2A_5
            \end{pmatrix}

        which will be expressed as :math:`RH_1 = D`.

        :math:`R` is almost a tridiagonal matrix, so we could use Thomas' algorithm.

        .. seealso::
            `Tridiagonal matrix algorithm. Wikipedia. <https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm>`_

        However, :math:`R` has ones at the opposite corners. A solution to this is
        the first decomposition proposed in the link below, with :math:`\alpha = 1`:

        .. seealso::
            `Tridiagonal matrix algorithm # Variants. Wikipedia. <https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm#Variants>`_

        .. math::
            R
            =
            \begin{pmatrix}
                4 & 1 & 0 & 0 & 1 \\
                1 & 4 & 1 & 0 & 0 \\
                0 & 1 & 4 & 1 & 0 \\
                0 & 0 & 1 & 4 & 1 \\
                1 & 0 & 0 & 1 & 4
            \end{pmatrix}
            &=
            \begin{pmatrix}
                3 & 1 & 0 & 0 & 0 \\
                1 & 4 & 1 & 0 & 0 \\
                0 & 1 & 4 & 1 & 0 \\
                0 & 0 & 1 & 4 & 1 \\
                0 & 0 & 0 & 1 & 3
            \end{pmatrix}
            +
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 1 \\
                0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 \\
                0 & 0 & 0 & 0 & 0 \\
                1 & 0 & 0 & 0 & 1
            \end{pmatrix}
            \\
            &
            \\
            &=
            \begin{pmatrix}
                3 & 1 & 0 & 0 & 0 \\
                1 & 4 & 1 & 0 & 0 \\
                0 & 1 & 4 & 1 & 0 \\
                0 & 0 & 1 & 4 & 1 \\
                0 & 0 & 0 & 1 & 3
            \end{pmatrix}
            +
            \begin{pmatrix}
                1 \\
                0 \\
                0 \\
                0 \\
                1
            \end{pmatrix}
            \begin{pmatrix}
                1 & 0 & 0 & 0 & 1
            \end{pmatrix}
            \\
            &
            \\
            &=
            T + uv^t

        We decompose :math:`R = T + uv^t`, where :math:`T` is a tridiagonal matrix, and
        :math:`u, v` are :math:`N`-D vectors such that :math:`u_0 = u_{N-1} = v_0 = v_{N-1} = 1`,
        and :math:`u_i = v_i = 0, \forall i \in \{1, ..., N-2\}`.

        Thus:

        .. math::
            RH_1 &= D \\
            \Rightarrow (T + uv^t)H_1 &= D

        If we find a vector :math:`q` such that :math:`Tq = u`:

        .. math::
            \Rightarrow (T + Tqv^t)H_1 &= D \\
            \Rightarrow T(I + qv^t)H_1 &= D \\
            \Rightarrow H_1 &= (I + qv^t)^{-1} T^{-1} D

        According to Sherman-Morrison's formula:

        .. seealso::
            `Sherman-Morrison's formula. Wikipedia. <https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula>`_

        .. math::
            (I + qv^t)^{-1} = I - \frac{1}{1 + v^tq} qv^t

        If we find :math:`Y = T^{-1} D`, or in other words, if we solve for
        :math:`Y` in :math:`TY = D`:

        .. math::
            H_1 &= (I + qv^t)^{-1} T^{-1} D \\
            &= (I + qv^t)^{-1} Y \\
            &= (I - \frac{1}{1 + v^tq} qv^t) Y \\
            &= Y - \frac{1}{1 + v^tq} qv^tY

        Therefore, we must solve for :math:`q` and :math:`Y` in :math:`Tq = u` and :math:`TY = D`.
        As :math:`T` is now tridiagonal, we shall use Thomas' algorithm.

        Define:

        *   :math:`a = [a_0, \ a_1, \ ..., \ a_{N-2}]` as :math:`T`'s lower diagonal of :math:`N-1` elements,
            such that :math:`a_0 = a_1 = ... = a_{N-2} = 1`, so this diagonal is filled with ones;
        *   :math:`b = [b_0, \ b_1, \ ..., \ b_{N-2}, \ b_{N-1}]` as :math:`T`'s main diagonal of :math:`N` elements,
            such that :math:`b_0 = b_{N-1} = 3`, and :math:`b_1 = b_2 = ... = b_{N-2} = 4`;
        *   :math:`c = [c_0, \ c_1, \ ..., \ c_{N-2}]` as :math:`T`'s upper diagonal of :math:`N-1` elements,
            such that :math:`c_0 = c_1 = ... = c_{N-2} = 1`: this diagonal is also filled with ones.

        If, according to Thomas' algorithm, we define:

        .. math::
            c'_0 &= \frac{c_0}{b_0} & \\
            c'_i &= \frac{c_i}{b_i - a_{i-1} c'_{i-1}}, & \quad \forall i \in \{1, ..., N-2\} \\
            & & \\
            u'_0 &= \frac{u_0}{b_0} & \\
            u'_i &= \frac{u_i - a_{i-1} u'_{i-1}}{b_i - a_{i-1} c'_{i-1}}, & \quad \forall i \in \{1, ..., N-1\} \\
            & & \\
            D'_0 &= \frac{1}{b_0} D_0 & \\
            D'_i &= \frac{1}{b_i - a_{i-1} c'_{i-1}} (D_i - a_{i-1} D'_{i-1}), & \quad \forall i \in \{1, ..., N-1\}

        Then:

        .. math::
            c'_0     &= \frac{1}{3} & \\
            c'_i     &= \frac{1}{4 - c'_{i-1}}, & \quad \forall i \in \{1, ..., N-2\} \\
            & & \\
            u'_0     &= \frac{1}{3} & \\
            u'_i     &= \frac{-u'_{i-1}}{4 - c'_{i-1}} = -c'_i u'_{i-1}, & \quad \forall i \in \{1, ..., N-2\} \\
            u'_{N-1} &= \frac{1 - u'_{N-2}}{3 - c'_{N-2}} & \\
            & & \\
            D'_0     &= \frac{1}{3} (4A_0 + 2A_1) & \\
            D'_i     &= \frac{1}{4 - c'_{i-1}} (4A_i + 2A_{i+1} - D'_{i-1}) & \\
            &= c_i (4A_i + 2A_{i+1} - D'_{i-1}), & \quad \forall i \in \{1, ..., N-2\} \\
            D'_{N-1} &= \frac{1}{3 - c'_{N-2}} (4A_{N-1} + 2A_N - D'_{N-2}) &

        Finally, we can do Backward Substitution to find :math:`q` and :math:`Y`:

        .. math::
            q_{N-1} &= u'_{N-1} & \\
            q_i     &= u'_{i} - c'_i q_{i+1}, & \quad \forall i \in \{0, ..., N-2\} \\
            & & \\
            Y_{N-1} &= D'_{N-1} & \\
            Y_i     &= D'_i - c'_i Y_{i+1},   & \quad \forall i \in \{0, ..., N-2\}

        With those values, we can finally calculate :math:`H_1 = Y - \frac{1}{1 + v^tq} qv^tY`.
        Given that :math:`v_0 = v_{N-1} = 1`, and :math:`v_1 = v_2 = ... = v_{N-2} = 0`, its dot products
        with :math:`q` and :math:`Y` are respectively :math:`v^tq = q_0 + q_{N-1}` and
        :math:`v^tY = Y_0 + Y_{N-1}`. Thus:

        .. math::
            H_1 = Y - \frac{1}{1 + q_0 + q_{N-1}} q(Y_0 + Y_{N-1})

        Once we have :math:`H_1`, we can get :math:`H_2` (the array of second handles) as follows:

        .. math::
            H_{2, i}   &= 2A_{i+1} - H_{1, i+1}, & \quad \forall i \in \{0, ..., N-2\} \\
            H_{2, N-1} &= 2A_0 - H_{1, 0} &

        Because the matrix :math:`R` always follows the same pattern (and thus :math:`T, u, v` as well),
        we can define a memo list for :math:`c'` and :math:`u'` to avoid recalculation. We cannot
        memoize :math:`D` and :math:`Y`, however, because they are always different matrices. We
        cannot make a memo for :math:`q` either, but we can calculate it faster because :math:`u'`
        can be memoized.

    Parameters
    ----------
    anchors
        Anchors of a closed cubic spline.

    Returns
    -------
    :class:`tuple` [:class:`~.Point3D_Array`, :class:`~.Point3D_Array`]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the closed cubic spline, and the other containing the 2nd handles.
    """
    global CP_CLOSED_MEMO
    global UP_CLOSED_MEMO

    A = np.asarray(anchors)
    N = A.shape[0] - 1
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
    # Calculate Dp (D prime).
    Dp = np.empty((N, dim))
    AUX = 4 * A[:N] + 2 * A[1:]  # Vectorize the sum for efficiency.
    Dp[0] = AUX[0] / 3
    for i in range(1, N - 1):
        Dp[i] = cp[i] * (AUX[i] - Dp[i - 1])
    Dp[N - 1] = cp_last_division * (AUX[N - 1] - Dp[N - 2])

    # Backward Substitution
    # Calculate Y, which is defined as a view of Dp for efficiency
    # and semantic convenience at the same time.
    Y = Dp
    # Y[N-1] = Dp[N-1] (redundant)
    for i in range(N - 2, -1, -1):
        Y[i] = Dp[i] - cp[i] * Y[i + 1]

    # Calculate H1.
    H1 = Y - 1 / (1 + q[0] + q[N - 1]) * q * (Y[0] + Y[N - 1])

    # Calculate H2.
    H2 = np.empty((N, dim))
    H2[0 : N - 1] = 2 * A[1:N] - H1[1:N]
    H2[N - 1] = 2 * A[N] - H1[0]

    return H1, H2


CP_OPEN_MEMO = np.array([0.5])


def get_smooth_open_cubic_bezier_handle_points(
    anchors: Point3DLike_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    r"""Special case of :func:`get_smooth_cubic_bezier_handle_points`,
    when the ``anchors`` do not form a closed loop.

    .. note::
        A system of equations must be solved to get the first handles of
        every Bèzier curve (referred to as :math:`H_1`).
        Then :math:`H_2` (the second handles) can be obtained separately.

        .. seealso::
            The equations were obtained from:

            * `Smooth Bézier Spline Through Prescribed Points. (2012). Particle in Cell Consulting LLC. <https://www.particleincell.com/2012/bezier-splines/>`_
            * `Conditions on control points for continuous curvature. (2016). Jaco Stuifbergen. <http://www.jacos.nl/jacos_html/spline/theory/theory_2.html>`_

        .. warning::
            The equations in the first webpage have some typos which were corrected in the comments.

        In general, if there are :math:`N+1` anchors, there will be :math:`N` Bézier curves
        and thus :math:`N` pairs of handles to find. We must solve the following
        system of equations for the 1st handles (example for :math:`N = 5`):

        .. math::
            \begin{pmatrix}
                2 & 1 & 0 & 0 & 0 \\
                1 & 4 & 1 & 0 & 0 \\
                0 & 1 & 4 & 1 & 0 \\
                0 & 0 & 1 & 4 & 1 \\
                0 & 0 & 0 & 2 & 7
            \end{pmatrix}
            \begin{pmatrix}
                H_{1,0} \\
                H_{1,1} \\
                H_{1,2} \\
                H_{1,3} \\
                H_{1,4}
            \end{pmatrix}
            =
            \begin{pmatrix}
                A_0 + 2A_1 \\
                4A_1 + 2A_2 \\
                4A_2 + 2A_3 \\
                4A_3 + 2A_4 \\
                8A_4 + A_5
            \end{pmatrix}

        which will be expressed as :math:`TH_1 = D`.
        :math:`T` is a tridiagonal matrix, so the system can be solved in :math:`O(N)`
        operations. Here we shall use Thomas' algorithm or the tridiagonal matrix
        algorithm.

        .. seealso::
            `Tridiagonal matrix algorithm. Wikipedia. <https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm>`_

        Define:

        *   :math:`a = [a_0, \ a_1, \ ..., \ a_{N-2}]` as :math:`T`'s lower diagonal of :math:`N-1` elements,
            such that :math:`a_0 = a_1 = ... = a_{N-3} = 1`, and :math:`a_{N-2} = 2`;
        *   :math:`b = [b_0, \ b_1, \ ..., \ b_{N-2}, \ b_{N-1}]` as :math:`T`'s main diagonal of :math:`N` elements,
            such that :math:`b_0 = 2`, :math:`b_1 = b_2 = ... = b_{N-2} = 4`, and :math:`b_{N-1} = 7`;
        *   :math:`c = [c_0, \ c_1, \ ..., \ c_{N-2}]` as :math:`T`'s upper diagonal of :math:`{N-1}` elements,
            such that :math:`c_0 = c_1 = ... = c_{N-2} = 1`: this diagonal is filled with ones.

        If, according to Thomas' algorithm, we define:

        .. math::
            c'_0 &= \frac{c_0}{b_0} & \\
            c'_i &= \frac{c_i}{b_i - a_{i-1} c'_{i-1}}, & \quad \forall i \in \{1, ..., N-2\} \\
            & & \\
            D'_0 &= \frac{1}{b_0} D_0 & \\
            D'_i &= \frac{1}{b_i - a_{i-1} c'{i-1}} (D_i - a_{i-1} D'_{i-1}), & \quad \forall i \in \{1, ..., N-1\}

        Then:

        .. math::
            c'_0     &= 0.5 & \\
            c'_i     &= \frac{1}{4 - c'_{i-1}}, & \quad \forall i \in \{1, ..., N-2\} \\
            & & \\
            D'_0     &= 0.5A_0 + A_1 & \\
            D'_i     &= \frac{1}{4 - c'_{i-1}} (4A_i + 2A_{i+1} - D'_{i-1}) & \\
            &= c_i (4A_i + 2A_{i+1} - D'_{i-1}), & \quad \forall i \in \{1, ..., N-2\} \\
            D'_{N-1} &= \frac{1}{7 - 2c'_{N-2}} (8A_{N-1} + A_N - 2D'_{N-2}) &

        Finally, we can do Backward Substitution to find :math:`H_1`:

        .. math::
            H_{1, N-1} &= D'_{N-1} & \\
            H_{1, i}   &= D'_i - c'_i H_{1, i+1}, & \quad \forall i \in \{0, ..., N-2\}

        Once we have :math:`H_1`, we can get :math:`H_2` (the array of second handles) as follows:

        .. math::
            H_{2, i}   &= 2A_{i+1} - H_{1, i+1}, & \quad \forall i \in \{0, ..., N-2\} \\
            H_{2, N-1} &= 0.5A_N   + 0.5H_{1, N-1} &

        As the matrix :math:`T` always follows the same pattern, we can define a memo list
        for :math:`c'` to avoid recalculation. We cannot do the same for :math:`D`, however,
        because it is always a different matrix.

    Parameters
    ----------
    anchors
        Anchors of an open cubic spline.

    Returns
    -------
    :class:`tuple` [:class:`~.Point3D_Array`, :class:`~.Point3D_Array`]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the open cubic spline, and the other containing the 2nd handles.
    """
    global CP_OPEN_MEMO

    A = np.asarray(anchors)
    N = A.shape[0] - 1
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

    # Calculate Dp (D prime).
    Dp = np.empty((N, dim))
    Dp[0] = 0.5 * A[0] + A[1]
    AUX = 4 * A[1 : N - 1] + 2 * A[2:N]  # Vectorize the sum for efficiency.
    for i in range(1, N - 1):
        Dp[i] = cp[i] * (AUX[i - 1] - Dp[i - 1])
    Dp[N - 1] = (1 / (7 - 2 * cp[N - 2])) * (8 * A[N - 1] + A[N] - 2 * Dp[N - 2])

    # Backward Substitution.
    # H1 (array of the first handles) is defined as a view of Dp for efficiency
    # and semantic convenience at the same time.
    H1 = Dp
    # H1[N-1] = Dp[N-1] (redundant)
    for i in range(N - 2, -1, -1):
        H1[i] = Dp[i] - cp[i] * H1[i + 1]

    # Calculate H2.
    H2 = np.empty((N, dim))
    H2[0 : N - 1] = 2 * A[1:N] - H1[1:N]
    H2[N - 1] = 0.5 * (A[N] + H1[N - 1])

    return H1, H2


@overload
def get_quadratic_approximation_of_cubic(
    a0: Point3DLike, h0: Point3DLike, h1: Point3DLike, a1: Point3DLike
) -> QuadraticSpline: ...


@overload
def get_quadratic_approximation_of_cubic(
    a0: Point3DLike_Array,
    h0: Point3DLike_Array,
    h1: Point3DLike_Array,
    a1: Point3DLike_Array,
) -> QuadraticBezierPath: ...


def get_quadratic_approximation_of_cubic(
    a0: Point3D | Point3D_Array,
    h0: Point3D | Point3D_Array,
    h1: Point3D | Point3D_Array,
    a1: Point3D | Point3D_Array,
) -> QuadraticSpline | QuadraticBezierPath:
    r"""If ``a0``, ``h0``, ``h1`` and ``a1`` are the control points of a cubic
    Bézier curve, approximate the curve with two quadratic Bézier curves and
    return an array of 6 points, where the first 3 points represent the first
    quadratic curve and the last 3 represent the second one.

    Otherwise, if ``a0``, ``h0``, ``h1`` and ``a1`` are _arrays_ of :math:`N`
    points representing :math:`N` cubic Bézier curves, return an array of
    :math:`6N` points where each group of :math:`6` consecutive points
    approximates each of the :math:`N` curves in a similar way as above.

    .. note::
        If the cubic spline given by the original cubic Bézier curves is
        smooth, this algorithm will generate a quadratic spline which is also
        smooth.

        If a cubic Bézier is given by

        .. math::
            C(t) = (1-t)^3 A_0 + 3(1-t)^2 t H_0 + 3(1-t)t^2 H_1 + t^3 A_1

        where :math:`A_0`, :math:`H_0`, :math:`H_1` and :math:`A_1` are its
        control points, then this algorithm should generate two quadratic
        Béziers given by

        .. math::
            Q_0(t) &= (1-t)^2 A_0 + 2(1-t)t M_0 + t^2 K \\
            Q_1(t) &= (1-t)^2 K + 2(1-t)t M_1 + t^2 A_1

        where :math:`M_0` and :math:`M_1` are the respective handles to be
        found for both curves, and :math:`K` is the end anchor of the 1st curve
        and the start anchor of the 2nd, which must also be found.

        To solve for :math:`M_0`, :math:`M_1` and :math:`K`, three conditions
        can be imposed:

        1.  :math:`Q_0'(0) = \frac{1}{2}C'(0)`. The derivative of the first
            quadratic curve at :math:`t = 0` should be proportional to that of
            the original cubic curve, also at :math:`t = 0`. Because the cubic
            curve is split into two parts, it is necessary to divide this by
            two: the speed of a point travelling through the curve should be
            half of the original. This gives:

            .. math::
                Q_0'(0) &= \frac{1}{2}C'(0) \\
                2(M_0 - A_0) &= \frac{3}{2}(H_0 - A_0) \\
                2M_0 - 2A_0 &= \frac{3}{2}H_0 - \frac{3}{2}A_0 \\
                2M_0 &= \frac{3}{2}H_0 + \frac{1}{2}A_0 \\
                M_0 &= \frac{1}{4}(3H_0 + A_0)

        2.  :math:`Q_1'(1) = \frac{1}{2}C'(1)`. The derivative of the second
            quadratic curve at :math:`t = 1` should be half of that of the
            original cubic curve for the same reasons as above, also at
            :math:`t = 1`. This gives:

            .. math::
                Q_1'(1) &= \frac{1}{2}C'(1) \\
                2(A_1 - M_1) &= \frac{3}{2}(A_1 - H_1) \\
                2A_1 - 2M_1 &= \frac{3}{2}A_1 - \frac{3}{2}H_1 \\
                -2M_1 &= -\frac{1}{2}A_1 - \frac{3}{2}H_1 \\
                M_1 &= \frac{1}{4}(3H_1 + A_1)

        3.  :math:`Q_0'(1) = Q_1'(0)`. The derivatives of both quadratic curves
            should match at the point :math:`K`, in order for the final spline
            to be smooth. This gives:

            .. math::
                Q_0'(1) &= Q_1'(0) \\
                2(K - M_0) &= 2(M_1 - K) \\
                2K - 2M_0 &= 2M_1 - 2K \\
                4K &= 2M_0 + 2M_1 \\
                K &= \frac{1}{2}(M_0 + M_1)

        This is sufficient to find proper control points for the quadratic
        Bézier curves.

    Parameters
    ----------
    a0
        The start anchor of a single cubic Bézier curve, or an array of
        :math:`N` start anchors for :math:`N` curves.
    h0
        The first handle of a single cubic Bézier curve, or an array of
        :math:`N` first handles for :math:`N` curves.
    h1
        The second handle of a single cubic Bézier curve, or an array of
        :math:`N` second handles for :math:`N` curves.
    a1
        The end anchor of a single cubic Bézier curve, or an array of
        :math:`N` end anchors for :math:`N` curves.

    Returns
    -------
    result
        An array containing either 6 points for 2 quadratic Bézier curves
        approximating the original cubic curve, or :math:`6N` points for
        :math:`2N` quadratic curves approximating :math:`N` cubic curves.

    Raises
    ------
    ValueError
        If ``a0``, ``h0``, ``h1`` and ``a1`` have different dimensions, or
        if their number of dimensions is not 1 or 2.
    """
    a0c = np.asarray(a0)
    h0c = np.asarray(h0)
    h1c = np.asarray(h1)
    a1c = np.asarray(a1)

    if all(arr.ndim == 1 for arr in (a0c, h0c, h1c, a1c)):
        num_curves, dim = 1, a0c.shape[0]
    elif all(arr.ndim == 2 for arr in (a0c, h0c, h1c, a1c)):
        num_curves, dim = a0c.shape
    else:
        raise ValueError("All arguments must be Point3D or Point3D_Array.")

    m0 = 0.25 * (3 * h0c + a0c)
    m1 = 0.25 * (3 * h1c + a1c)
    k = 0.5 * (m0 + m1)

    result = np.empty((6 * num_curves, dim))
    result[0::6] = a0c
    result[1::6] = m0
    result[2::6] = k
    result[3::6] = k
    result[4::6] = m1
    result[5::6] = a1c
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
    return bool(abs(end[2] - start[2]) <= tolerance[2])


def proportions_along_bezier_curve_for_point(
    point: Point3DLike,
    control_points: BezierPointsLike,
    round_to: float = 1e-6,
) -> MatrixMN:
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
    # Get common roots
    # arg-type: ignore
    roots = reduce(np.intersect1d, roots)
    result = np.asarray([r.real for r in roots if 0 <= r.real <= 1])
    return result


def point_lies_on_bezier(
    point: Point3DLike,
    control_points: BezierPointsLike,
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
