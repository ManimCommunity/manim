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
    "partial_quadratic_bezier_points",
    "interpolate",
    "integer_interpolate",
    "mid",
    "inverse_interpolate",
    "match_interpolate",
    "get_smooth_handle_points",
    "get_smooth_cubic_bezier_handle_points",
    "is_closed",
    "proportions_along_bezier_curve_for_point",
    "point_lies_on_bezier",
]


from functools import reduce
from typing import Any, Callable, Sequence, overload

import numpy as np
import numpy.typing as npt

from manim.utils.simple_functions import get_pascal_triangle
from manim.utils.space_ops import cross2d, find_intersection


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

    # Combinatorial coefficients
    choose_n = get_pascal_triangle(n)[n]

    return lambda t: np.asarray(
        np.asarray(
            [
                (((1 - t) ** (n - k)) * (t**k) * choose_n[k] * point)
                for k, point in enumerate(points)
            ],
            dtype=PointDType,
        ).sum(axis=0)
    )


# !TODO: This function has still a weird implementation with the overlapping points
def partial_bezier_points(points: BezierPoints, a: float, b: float) -> BezierPoints:
    """Given an array of points which define bezier curve, and two numbers 0<=a<b<=1, return an array of the same size,
    which describes the portion of the original bezier curve on the interval [a, b].

    This algorithm is pretty nifty, and pretty dense.

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
    _len = len(points)
    if a == 1:
        return np.asarray([points[-1]] * _len, dtype=PointDType)

    a_to_1 = np.asarray(
        [bezier(points[i:])(a) for i in range(_len)],
        dtype=PointDType,
    )
    end_prop = (b - a) / (1.0 - a)
    return np.asarray(
        [bezier(a_to_1[: i + 1])(end_prop) for i in range(_len)],
        dtype=PointDType,
    )


# Shortened version of partial_bezier_points just for quadratics,
# since this is called a fair amount
def partial_quadratic_bezier_points(
    points: QuadraticBezierPoints, a: float, b: float
) -> QuadraticBezierPoints:
    if a == 1:
        return np.asarray(3 * [points[-1]])

    def curve(t: float) -> Point3D:
        return np.asarray(
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
    return np.asarray((h0, h1, h2))


def split_quadratic_bezier(points: QuadraticBezierPoints, t: float) -> BezierPoints:
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

    return np.array((a1, s1, p, p, s2, a2))


def subdivide_quadratic_bezier(points: QuadraticBezierPoints, n: int) -> BezierPoints:
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
def interpolate(start: float, end: float, alpha: float) -> float:
    ...


@overload
def interpolate(start: Point3D, end: Point3D, alpha: float) -> Point3D:
    ...


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


# Figuring out which Bézier curves most smoothly connect a sequence of points
def get_smooth_handle_points(
    anchors: Point3D_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
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
    :class:`tuple` [:class:`~.Point3D_Array`, :class:`~.Point3D_Array`]
        A tuple of two arrays: one containing the 1st handle for every curve in
        the cubic spline, and the other containing the 2nd handles.
    """
    anchors = np.asarray(anchors)
    n_handles = len(anchors) - 1

    # If there's a single anchor, there's no Bézier curve.
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
        return get_handles_for_smooth_closed_cubic_spline(anchors)
    else:
        return get_handles_for_smooth_open_cubic_spline(anchors)


CP_CLOSED_MEMO = np.array([1 / 3])
UP_CLOSED_MEMO = np.array([1 / 3])


def get_handles_for_smooth_closed_cubic_spline(
    anchors: Point3D_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    r"""Special case of :func:`get_handles_for_smooth_cubic_spline`,
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
                0
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


def get_handles_for_smooth_open_cubic_spline(
    anchors: Point3D_Array,
) -> tuple[Point3D_Array, Point3D_Array]:
    r"""Special case of :func:`get_handles_for_smooth_cubic_spline`,
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
    return np.allclose(points[0], points[-1])  # type: ignore


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

    # Combinatorial coefficients
    choose = get_pascal_triangle(n)
    choose_n = choose[n]

    roots = []
    for dim, coord in enumerate(point):
        control_coords = control_points[:, dim]
        terms = []
        for term_power in range(n, -1, -1):
            outercoeff = choose_n[term_power]
            term = []
            sign = 1
            choose_pow = choose[term_power]
            for subterm_num in range(term_power, -1, -1):
                innercoeff = choose_pow[subterm_num] * sign
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
