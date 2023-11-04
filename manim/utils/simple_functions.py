"""A collection of simple functions."""

from __future__ import annotations

__all__ = [
    "binary_search",
    "choose",
    "clip",
    "sigmoid",
]


import inspect
from functools import lru_cache
from types import MappingProxyType
from typing import Callable

import numpy as np
from scipy import special


def binary_search(
    function: Callable[[int | float], int | float],
    target: int | float,
    lower_bound: int | float,
    upper_bound: int | float,
    tolerance: int | float = 1e-4,
) -> int | float | None:
    """Searches for a value in a range by repeatedly dividing the range in half.

    To be more precise, performs numerical binary search to determine the
    input to ``function``, between the bounds given, that outputs ``target``
    to within ``tolerance`` (default of 0.0001).
    Returns ``None`` if no input can be found within the bounds.

    Examples
    --------

    Consider the polynomial :math:`x^2 + 3x + 1` where we search for
    a target value of :math:`11`. An exact solution is :math:`x = 2`.

    ::

        >>> solution = binary_search(lambda x: x**2 + 3*x + 1, 11, 0, 5)
        >>> abs(solution - 2) < 1e-4
        True
        >>> solution = binary_search(lambda x: x**2 + 3*x + 1, 11, 0, 5, tolerance=0.01)
        >>> abs(solution - 2) < 0.01
        True

    Searching in the interval :math:`[0, 5]` for a target value of :math:`71`
    does not yield a solution::

        >>> binary_search(lambda x: x**2 + 3*x + 1, 71, 0, 5) is None
        True
    """
    lh = lower_bound
    rh = upper_bound
    mh = np.mean(np.array([lh, rh]))
    while abs(rh - lh) > tolerance:
        mh = np.mean(np.array([lh, rh]))
        lx, mx, rx = (function(h) for h in (lh, mh, rh))
        if lx == target:
            return lh
        if rx == target:
            return rh

        if lx <= target <= rx:
            if mx > target:
                rh = mh
            else:
                lh = mh
        elif lx > target > rx:
            lh, rh = rh, lh
        else:
            return None

    return mh


@lru_cache(maxsize=10)
def choose(n: int, k: int) -> int:
    r"""The binomial coefficient n choose k.

    :math:`\binom{n}{k}` describes the number of possible choices of
    :math:`k` elements from a set of :math:`n` elements.

    References
    ----------
    - https://en.wikipedia.org/wiki/Combination
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html
    """
    return special.comb(n, k, exact=True)


def clip(a, min_a, max_a):
    """Clips ``a`` to the interval [``min_a``, ``max_a``].

    Accepts any comparable objects (i.e. those that support <, >).
    Returns ``a`` if it is between ``min_a`` and ``max_a``.
    Otherwise, whichever of ``min_a`` and ``max_a`` is closest.

    Examples
    --------
    ::

        >>> clip(15, 11, 20)
        15
        >>> clip('a', 'h', 'k')
        'h'
    """
    if a < min_a:
        return min_a
    elif a > max_a:
        return max_a
    return a


def sigmoid(x: float) -> float:
    r"""Returns the output of the logistic function.

    The logistic function, a common example of a sigmoid function, is defined
    as :math:`\frac{1}{1 + e^{-x}}`.

    References
    ----------
    - https://en.wikipedia.org/wiki/Sigmoid_function
    - https://en.wikipedia.org/wiki/Logistic_function
    """
    return 1.0 / (1 + np.exp(-x))
