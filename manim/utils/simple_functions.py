"""A collection of simple functions."""

from __future__ import annotations

__all__ = [
    "binary_search",
    "choose",
    "clip",
    "get_parameters",
    "sigmoid",
]


import inspect
from functools import lru_cache
from types import MappingProxyType
from typing import Callable, Union

import numpy as np
from scipy import special


def binary_search(
    function: Callable[[int | float], int | float],
    target: int | float,
    lower_bound: int | float,
    upper_bound: int | float,
    tolerance: int | float = 1e-4,
) -> int | float | None:
    """Performs a numerical binary search to determine the input to `function`,
    between the bounds given, that outputs `target` to within `tolerance` (default of 0.0001).
    Returns None if no input can be found within the bounds.

    Examples
    --------
    Normal Usage::

        Observe that 0 <= 2 (solution) <= 5 and (2)^2 + 3(2) + 1 = 11 (target)
        >>>binary_search(lambda x: x**2 + 3*x + 1, 11, 0, 5)
        2.0000457763671875
        >>>binary_search(lambda x: x**2 + 3*x + 1, 11, 0, 5, 0.01)
        2.001953125

        Here, observe that 7 (solution) > 5 (upper_bound) and (7)^2 + 3(7) + 1 = 71 (target)
        >>>binary_search(lambda x: x**2 + 3*x + 1, 71, 0, 5)
        None
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
    """'n choose k' - the number of combinations of `n` things taken `k` at a time.

    References
    ----------
    - https://en.wikipedia.org/wiki/Combination
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html
    """
    return special.comb(n, k, exact=True)


def clip(a, min_a, max_a):
    """Accepts any comparable objects (i.e. those that support <, >).
    Returns `a` if it is between `min_a` and `max_a`.
    Otherwise, whichever of `min_a` and `max_a` is closest.

    Examples
    --------
    Normal Usage::

        >>>clip(15, 11, 20)
        15

        >>>clip('a', 'h', 'k')
        'h'
    """
    if a < min_a:
        return min_a
    elif a > max_a:
        return max_a
    return a


def get_parameters(function: Callable) -> MappingProxyType[str, inspect.Parameter]:
    """Return the parameters of `function` as an ordered mapping of parameters'
    names to their corresponding `Parameter` objects.

    Examples
    --------
    ::

        >>> get_parameters(get_parameters)
        mappingproxy(OrderedDict([('function', <Parameter "function: 'Callable'">)]))

        >>> dict(get_parameters(choose))
        {'n': <Parameter "n: 'int'">, 'k': <Parameter "k: 'int'">}
    """
    return inspect.signature(function).parameters


def sigmoid(x: float) -> float:
    """Returns the output of the logistic function (a very common sigmoid
    function) defined as 1/(1+e^-x)

    References
    ----------
    - https://en.wikipedia.org/wiki/Sigmoid_function
    - https://en.wikipedia.org/wiki/Logistic_function
    """
    return 1.0 / (1 + np.exp(-x))
