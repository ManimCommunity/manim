"""Utility functions related to Bézier curves."""

from __future__ import annotations

__all__ = ["ManimBezier"]

import sys
import typing
from functools import reduce
from typing import Callable, Dict, Iterable

import numpy as np

from ..utils.simple_functions import choose
from ..utils.space_ops import cross2d, find_intersection


class ManimBezier:
    # Placeholder value
    degree = 1

    # Memo for get_function_from_points
    negative_pascal_triangle = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, -1, 0, 0, 0, 0],
            [1, -2, 1, 0, 0, 0],
            [1, -3, 3, -1, 0, 0],
            [1, -4, 6, -4, 1, 0],
            [1, -5, 10, -10, 5, -1],
        ]
    )

    # Memos for get_smooth_cubic_bezier_handles_for_(open/closed)_curve
    _cp_open_memo = np.array([0.5])
    _cp_closed_memo = np.array([1 / 3])
    _up_closed_memo = np.array([1 / 3])

    # Memo for subdivide_bezier, to be redefined in subclasses
    subdivision_matrices: dict[int, np.ndarray] = {}

    tolerance_for_point_equality = 1e-8

    def __new__(cls, *args, **kwargs):
        if cls.__name__ == "ManimBezier":
            raise NotImplementedError(
                f"The class {cls.__name__} is not intended to be "
                f"instantiated directly. Only its subclasses with the class "
                f"attribute degree should be instantiated."
            )

        return super().__new__(cls)

    def __init__(
        self,
        points: np.ndarray | None = None,
        dim: int = 3,
    ):
        if points is None:
            points = np.empty((0, dim))
        self._internal = points
        self.dim = dim

        # LUT (Look-Up Table)-related attributes
        self._num_segments_per_curve = 10
        nspc = self._num_segments_per_curve
        self._t_range = (np.arange(nspc + 1) / nspc).reshape(-1, 1)

        self._lut = np.empty((0, nspc + 1, dim))
        self._arc_length_pieces = np.empty((0, nspc))
        self._accumulated_arc_length_pieces = np.empty((0, nspc))
        self._arc_lengths = np.empty(0)
        self._total_arc_length = 0.0

        self.needs_new_lut = True
        self.needs_new_arc_length_pieces = True
        self.needs_new_accumulated_arc_length_pieces = True
        self.needs_new_arc_lengths = True
        self.needs_new_total_arc_length = True

    def __add__(self, shift_vector: np.ndarray):
        shift_vector = np.asarray(shift_vector)
        self._internal += shift_vector
        if shift_vector.ndim > 1:
            self.needs_new_lut = True
            self.needs_new_arc_length_pieces = True
            self.needs_new_accumulated_arc_length_pieces = True
            self.needs_new_arc_lengths = True
            self.needs_new_total_arc_length = True
        elif not self.needs_new_lut:
            self._lut += shift_vector
        return self

    def __sub__(self, shift_vector: np.ndarray):
        return self.__add__(-np.asarray(shift_vector))

    def __mul__(self, factor: float | np.ndarray):
        factor_as_arr = np.asarray(factor)
        if factor_as_arr.ndim == 1:
            self._internal *= factor
            if not self.needs_new_lut:
                self._lut *= factor
            if not self.needs_new_arc_length_pieces:
                self._arc_length_pieces *= factor
            if not self.needs_new_accumulated_arc_length_pieces:
                self._accumulated_arc_length_pieces *= factor
            if not self.needs_new_arc_lengths:
                self._arc_lengths *= factor
            if not self.needs_new_total_arc_length:
                self._total_arc_length *= factor
        else:
            self._internal *= factor_as_arr
            self.needs_new_lut = True
            self.needs_new_arc_length_pieces = True
            self.needs_new_accumulated_arc_length_pieces = True
            self.needs_new_arc_lengths = True
            self.needs_new_total_arc_length = True

        return self

    def __div__(self, factor: float | np.ndarray):
        if type(factor) == float:
            return self.__mul__(1 / factor)
        return self.__mul__(1 / np.asarray(factor))

    def __array__(self):
        return self._internal

    def __ufunc__(self, method, *args, **kwargs):
        return self.__class__(method(self._internal), *args, **kwargs)

    # Properties and methods related to the LUT (Look-Up Table) and arc lengths
    @property
    def num_segments_per_curve(self):
        return self._num_segments_per_curve

    @num_segments_per_curve.setter
    def num_segments_per_curve(self, new_value):
        if new_value != self._num_segments_per_curve:
            self._num_segments_per_curve = new_value
            self._t_range = (np.arange(new_value + 1) / new_value).reshape(-1, 1)
            self.needs_new_lut = True
            self.needs_new_arc_length_pieces = True
            self.needs_new_accumulated_arc_length_pieces = True
            self.needs_new_arc_lengths = True
            self.needs_new_total_arc_length = True

    @property
    def nspc(self):
        return self._num_segments_per_curve

    @nspc.setter
    def nspc(self, new_value):
        self._num_segments_per_curve = new_value

    @property
    def lut(self):
        if self.degree is None:
            raise NotImplementedError(
                f"A LUT can only be calculated from subclasses of "
                f"{self.__class__.__name__} with a defined degree attribute"
            )

        if self.needs_new_lut:
            num_points = self._internal.shape[0]
            nppc = self.degree + 1  # Number of Points Per Curve
            assert num_points % nppc == 0
            num_curves = num_points // nppc

            self._lut = np.empty((num_curves, self.nspc + 1, self.dim))
            for i in range(num_curves):
                bezier_function = self.get_function_from_points(
                    self._internal[i * nppc : (i + 1) * nppc]
                )
                self._lut[i : i + 1] = bezier_function(self._t_range)

        return self._lut

    @property
    def arc_length_pieces(self):
        if self.needs_new_arc_length_pieces:
            # This triggers calculation of the LUT
            # if it is not already calculated
            lut = self.lut
            diffs = lut[:, 1:] - lut[:, :-1]
            square_dists = np.sum(diffs * diffs, axis=2)
            self._arc_length_pieces = np.sqrt(square_dists)
            self.needs_new_arc_length_pieces = False

        return self._arc_length_pieces

    @property
    def accumulated_arc_length_pieces(self):
        if self.needs_new_accumulated_arc_length_pieces:
            # This triggers calculation of the arc length pieces
            # if they are not already calculated
            arc_length_pieces = self.arc_length_pieces
            self._accumulated_arc_length_pieces = np.add.accumulate(
                arc_length_pieces, axis=1
            )
            self.needs_new_accumulated_arc_length_pieces = False

        return self._accumulated_arc_length_pieces

    @property
    def arc_lengths(self):
        if self.needs_new_arc_lengths:
            # This triggers calculation of the arc length pieces
            # if they are not already calculated
            arc_length_pieces = self.arc_length_pieces
            self._arc_lengths = np.sum(arc_length_pieces, axis=1)
            self.needs_new_arc_lengths = False

        return self._arc_lengths

    @property
    def total_arc_length(self):
        if self.needs_new_total_arc_length:
            # This triggers calculation of the arc lengths
            # if they are not already calculated
            arc_lengths = self.arc_lengths
            self._total_arc_length = np.add.accumulate(arc_lengths)[-1]
            self.needs_new_total_arc_length = False

        return self._total_arc_length

    @property
    def num_curves(self):
        return self._internal.shape[0] / (self.degree + 1)

    def get_nth_curve_points(self, n: int) -> np.ndarray:
        nppc = self.degree + 1
        return self._internal[nppc * n : nppc * (n + 1)]

    def get_nth_curve_function(self, n: int) -> Callable[[float], np.ndarray]:
        nppc = self.degree + 1
        return self.get_function_from_points(self._internal[nppc * n : nppc * (n + 1)])

    def get_nth_curve_length_pieces(
        self, n: int, num_segments_per_curve: int | None = None
    ) -> float:
        if num_segments_per_curve is None:
            num_segments_per_curve = 10
        self.num_segments_per_curve = num_segments_per_curve
        arc_length_pieces = self.arc_length_pieces
        return arc_length_pieces[n]

    def get_nth_curve_length(self, n: int) -> float:
        arc_lengths = self.arc_lengths
        return arc_lengths[n]

    def get_nth_curve_function_with_length(
        self,
        n: int,
    ) -> tuple[Callable[[float], np.ndarray], float]:
        return (self.get_nth_curve_function(n), self.get_nth_curve_length(n))

    def get_curve_functions(
        self,
    ) -> Iterable[Callable[[float], np.ndarray]]:
        """Gets the functions for the curves of the mobject.

        Returns
        -------
        typing.Iterable[typing.Callable[[float], np.ndarray]]
            The functions for the curves.
        """

        num_curves = self.num_curves

        for n in range(num_curves):
            yield self.get_nth_curve_function(n)

    # Bezier to Function
    @classmethod
    def get_function_from_points(
        cls,
        points: np.ndarray,
    ) -> Callable[[float], np.ndarray]:
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
        n = len(points) - 1

        # Cubic Bézier curve
        if n == 3:
            C0 = points[0]
            C1 = 3 * (points[1] - points[0])
            C2 = 3 * (points[2] - 2 * points[1] + points[0])
            C3 = points[3] - 3 * points[2] + 3 * points[1] - points[0]

            def cubic_bezier(t):
                return C0 + t * (C1 + t * (C2 + t * C3))

            return cubic_bezier

        # Quadratic Bézier curve
        if n == 2:
            C0 = points[0]
            C1 = 2 * (points[1] - points[0])
            C2 = points[2] - 2 * points[1] + points[0]

            def quadratic_bezier(t):
                return C0 + t * (C1 + t * C2)

            return quadratic_bezier

        # Linear Bézier curve
        if n == 1:
            C0 = points[0]
            C1 = points[1] - points[0]

            def linear_bezier(t):
                return C0 + t * C1

            return linear_bezier

        # Degree 0 Bézier curve
        if n == 0:

            def degree_zero_bezier(t):
                return points[0]

            return degree_zero_bezier

        # Degree n Bézier curve
        # Memoize coefficients for later use
        old_n = cls.negative_pascal_triangle.shape[0] - 1
        if n > old_n:
            new_triangle = np.zeros((n + 1, n + 1))
            new_triangle[: old_n + 1, : old_n + 1] = cls.negative_pascal_triangle
            for i in range(old_n + 1, n + 1):
                new_triangle[i, : i + 1] = new_triangle[i - 1, :i]
                new_triangle[i, 1 : i + 2] -= new_triangle[i - 1, :i]
            cls.negative_pascal_triangle = new_triangle

        neg_binomials = cls.negative_pascal_triangle[: n + 1, : n + 1]
        C = np.dot(neg_binomials, points) * neg_binomials[-1:].T
        exponents = np.arange(n + 1)

        def degree_n_bezier(t):
            return np.dot(t**exponents, C)

        return degree_n_bezier

    def make_jagged(self):
        raise NotImplementedError

    def make_smooth(self):
        raise NotImplementedError

    def make_approximately_smooth(self):
        raise NotImplementedError

    def change_anchor_mode(self, mode: str):
        """Changes the anchor mode of the bezier curves. This will modify the handles.

        There can be only two modes, "jagged", and "smooth".

        Returns
        -------
        :class:`ManimBezier`
            ``self``
        """
        if mode == "jagged":
            return self.make_jagged()
        if mode in ["smooth", "true_smooth"]:
            return self.make_smooth()
        if mode == "approx_smooth":
            return self.make_approximately_smooth()
        raise ValueError(
            "Mode must be 'jagged', 'smooth'/'true_smooth' or 'approx_smooth'."
        )

    # Helper functions for make_smooth and make_approximately_smooth,
    # intended to be used in subclasses
    @staticmethod
    def get_smooth_quadratic_bezier_handle_points(anchors: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def get_smooth_cubic_bezier_handle_points(
        cls,
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
        n_handles = anchors.shape[0] - 1

        # If there's a single anchor, there's no Bezier curve.
        # Return empty arrays.
        if n_handles == 0:
            dim = anchors.shape[1]
            return np.zeros((0, dim)), np.zeros((0, dim))

        # If there are only two anchors (thus only one pair of handles),
        # they can only be an interpolation of these two anchors with alphas
        # 1/3 and 2/3, which will draw a straight line between the anchors.
        if n_handles == 1:
            h1 = (2 * anchors[0] + anchors[1]) / 3
            h2 = (anchors[0] + 2 * anchors[1]) / 3
            return h1, h2

        # Handle different cases depending on whether the points form a closed
        # curve or not
        curve_is_closed = cls.is_closed(anchors)
        if curve_is_closed:
            return cls._smooth_closed_cubic(anchors)
        else:
            return cls._smooth_open_cubic(anchors)

    @classmethod
    def _smooth_closed_cubic(
        cls,
        anchors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        N = A.shape[0] - 1
        dim = A.shape[1]

        # Calculate cp (c prime) and up (u prime) with help from
        # CP_CLOSED_MEMO and UP_CLOSED_MEMO.
        len_memo = cls._cp_closed_memo.size
        if len_memo < N - 1:
            cp = np.empty(N - 1)
            up = np.empty(N - 1)
            cp[:len_memo] = cls._cp_closed_memo
            up[:len_memo] = cls._up_closed_memo
            # Forward Substitution 1
            # Calculate up (at the same time we calculate cp).
            for i in range(len_memo, N - 1):
                cp[i] = 1 / (4 - cp[i - 1])
                up[i] = -cp[i] * up[i - 1]
            cls._cp_closed_memo = cp
            cls._up_closed_memo = up
        else:
            cp = cls._cp_closed_memo[: N - 1]
            up = cls._up_closed_memo[: N - 1]

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

    @classmethod
    def _smooth_open_cubic(
        cls,
        anchors: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        N = A.shape[0] - 1
        dim = A.shape[1]

        # Calculate cp (c prime) with help from CP_OPEN_MEMO.
        len_memo = cls._cp_open_memo.size
        if len_memo < N - 1:
            cp = np.empty(N - 1)
            cp[:len_memo] = cls._cp_open_memo
            for i in range(len_memo, N - 1):
                cp[i] = 1 / (4 - cp[i - 1])
            cls._cp_open_memo = cp
        else:
            cp = cls._cp_open_memo[: N - 1]

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

    # Given 4 control points for a cubic bezier curve (or arrays of such)
    # return control points for 2 quadratics (or 2n quadratics) approximating them.
    @classmethod
    def get_quadratic_approximation_of_cubic(
        cls,
        a0: np.ndarray,
        h0: np.ndarray,
        h1: np.ndarray,
        a1: np.ndarray,
    ) -> np.ndarray:
        a0 = np.asarray(a0, ndmin=2)
        h0 = np.asarray(h0, ndmin=2)
        h1 = np.asarray(h1, ndmin=2)
        a1 = np.asarray(a1, ndmin=2)
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

        t_mid = 0.5 * np.ones(len(a0))
        t_mid[ti_min_in_range] = ti_min[ti_min_in_range]
        t_mid[ti_max_in_range] = ti_max[ti_max_in_range]

        m, n = a0.shape
        t_mid = t_mid.repeat(n).reshape((m, n))

        # Compute bezier point and tangent at the chosen value of t
        mid = cls.get_function_from_points([a0, h0, h1, a1])(t_mid)
        Tm = cls.get_function_from_points([h0 - a0, h1 - h0, a1 - h1])(t_mid)

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

    @classmethod
    def is_closed(cls, points: tuple[np.ndarray, np.ndarray]) -> bool:
        """Returns True if the curve given by the points is closed, by checking if its
        first and last points are close to each other.

        This function reimplements np.allclose (without a relative tolerance rtol),
        because repeated calling of np.allclose for only 2 points is inefficient.
        """
        start, end = points[0], points[-1]
        atol = cls.tolerance_for_point_equality
        if abs(end[0] - start[0]) > atol:
            return False
        if abs(end[1] - start[1]) > atol:
            return False
        if abs(end[2] - start[2]) > atol:
            return False
        return True

    def proportion_from_point(
        self,
        point: np.ndarray,
        round_to: float | int | None = 1e-6,
    ) -> np.ndarray:
        """Returns the proportion along the path of the :class:`VMobject`
        a particular given point is at.

        Parameters
        ----------
        point
            The Cartesian coordinates of the point which may or may not lie on the :class:`VMobject`

        Returns
        -------
        float
            The proportion along the path of the :class:`VMobject`.

        Raises
        ------
        :exc:`ValueError`
            If ``point`` does not lie on the curve.
        :exc:`Exception`
            If the :class:`VMobject` has no points.
        """
        point = np.asarray(point)
        square_atol = 1e-15

        lut = self.lut
        arc_length_pieces = self.arc_length_pieces
        acc_arc_length_pieces = self.accumulated_arc_length_pieces
        total_arc_length = self.total_arc_length

        num_curves = self.num_curves
        num_samples = num_curves * self.nspc + 1
        vectors = np.empty((num_samples, self.dim))
        vectors[0] = lut[0, 0] - point
        vectors[1:] = lut[:, 1:].flat - point

        initial_square_dists = np.sum(vectors * vectors, axis=1)
        min_dist_index = np.argmin(initial_square_dists)
        curr_min = initial_square_dists[min_dist_index]

        curve_index = min_dist_index // self.nspc
        step = 0.5 / self.nspc

        if min_dist_index == 0:
            curve_function = self.get_nth_curve_function(0)
            left = curr_min
            right = initial_square_dists[1]
            mid = curve_function(step)
            curve_t = step

        elif min_dist_index == num_samples - 1:
            curve_function = self.get_nth_curve_function(num_curves - 1)
            left = initial_square_dists[-2]
            right = curr_min
            mid = curve_function(1 - step)
            curve_index -= 1
            curve_t = 1 - step

        else:
            right_func = self.get_nth_curve_function(curve_index)
            if min_dist_index % self.nspc == 0:
                left_func = self.get_nth_curve_function(curve_index - 1)
            else:
                left_func = right_func

            mid_left = left_func(1 - step)
            mid_left = np.sum(mid_left * mid_left)
            mid_right = right_func(step)
            mid_right = np.sum(mid_right * mid_right)
            mid = curr_min
            if mid - mid_left > square_atol:
                if mid - mid_right > square_atol:
                    if mid > square_atol:
                        raise ValueError(f"Point {point} does not lie on this curve.")
                    return acc_arc_length_pieces[min_dist_index - 1] / total_arc_length
                else:
                    curve_function = left_func
                    left = initial_square_dists[min_dist_index - 1]
                    right = mid
                    mid = mid_left
                    curve_index -= 1
                    curve_t = 1 - step
            else:
                curve_function = right_func
                left = mid
                right = initial_square_dists[min_dist_index + 1]
                mid = mid_right
                curve_t = step

        prev_min = curr_min
        curr_min = mid
        step /= 2

        while prev_min - curr_min > square_atol:
            mid_left = curve_function(curve_t - step)
            mid_left = np.sum(mid_left * mid_left)
            mid_right = curve_function(curve_t + step)
            mid_right = np.sum(mid_right * mid_right)
            mid = curr_min
            if mid - mid_left > square_atol:
                if mid - mid_right > square_atol:
                    if mid > square_atol:
                        raise ValueError(f"Point {point} does not lie on this curve.")
                else:
                    # left = left
                    right = mid
                    mid = mid_left
                    curve_t -= step
            else:
                left = mid
                # right = right
                mid = mid_right
                curve_t += step

            prev_min = curr_min
            curr_min = mid
            step /= 2

        return

    @staticmethod
    def partial_bezier_points(points: np.ndarray, a: float, b: float) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def split_bezier(points: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def subdivide_bezier(cls, points: Iterable[float], n: int) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def get_bezier_tuples_from_points(cls, points):
        nppc = cls.degree + 1
        points = np.asarray(points)
        num_points, dim = points.shape
        num_curves = num_points // nppc
        return points[: num_curves * nppc].reshape(-1, nppc, dim)

    def get_bezier_tuples(self):
        return self.get_bezier_tuples_from_points(self._internal)

    @classmethod
    def insert_n_curves_to_point_list(cls, n: int, points: np.ndarray) -> np.ndarray:
        """Given an array of k points defining Bézier curves (anchors and handles), returns points defining exactly k + n Bézier curves.

        Parameters
        ----------
        n
            Number of desired curves.
        points
            Starting points.

        Returns
        -------
        np.ndarray
            Points generated.
        """
        points = np.asarray(points)
        if points.shape[0] == 1:
            return np.repeat(points, n * (cls.degree + 1), 0)

        bezier_tuples = cls.get_bezier_tuples_from_points(points)
        curr_num_curves, nppc, dim = bezier_tuples.shape
        target_num_curves = curr_num_curves + n
        # This is an array with values ranging from 0
        # up to curr_num_curves,  with repeats such that
        # it's total length is target_num_curves.  For example,
        # with curr_num_curves = 10, target_num_curves = 15, this
        # would be [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9].
        repeat_indices = (
            np.arange(target_num_curves, dtype="i") * curr_num_curves
        ) // target_num_curves

        # If the nth term of this list is k, it means
        # that the nth curve of our path should be split
        # into k pieces.
        # In the above example our array had the following elements
        # [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9]
        # We have two 0s, one 1, two 2s and so on.
        # The split factors array would hence be:
        # [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        split_factors = np.zeros(curr_num_curves, dtype="i")
        np.add.at(split_factors, repeat_indices, 1)

        new_points = np.empty((target_num_curves, dim))
        index = 0
        for curve, sf in zip(bezier_tuples, split_factors):
            if sf == 1:
                new_points[index] = curve
            else:
                new_points[index : index + sf] = cls.subdivide_bezier(curve, sf)
            index += sf

        return new_points

    @classmethod
    def bezier_remap(
        cls,
        curves: np.ndarray,
        target_num_curves: int,
    ) -> np.ndarray:
        """Subdivides each curve in curves into as many parts as necessary, until the final number of curves reaches a desired amount, target_num_curves.

        Parameters
        ----------
        curves
            An array of n Bézier curves to be remapped. The shape of this array must be (current_num_curves, degree+1, dimension).

        target_num_curves
            The number of curves that the output will contain. This needs to be higher than the current number.

        Returns
        -------
            The new Bézier curves after the remap.
        """
        curves = np.asarray(curves)
        # NPPC: Number of Points Per Curve = degree + 1
        current_num_curves, nppc, dim = curves.shape
        num_curves_to_insert = target_num_curves - current_num_curves
        if num_curves_to_insert <= 0:
            return curves

        return cls.insert_n_curves_to_point_list(
            curves.flat, num_curves_to_insert
        ).reshape(target_num_curves, nppc, dim)
