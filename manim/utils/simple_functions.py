"""A collection of simple functions."""

from __future__ import annotations

__all__ = [
    "sigmoid",
    "choose",
    "get_parameters",
    "binary_search",
]


import inspect
from functools import lru_cache

import numpy as np
from scipy import special


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


@lru_cache(maxsize=10)
def choose(n, k):
    return special.comb(n, k, exact=True)


def get_parameters(function):
    return inspect.signature(function).parameters


def clip(a, min_a, max_a):
    if a < min_a:
        return min_a
    elif a > max_a:
        return max_a
    return a


def binary_search(function, target, lower_bound, upper_bound, tolerance=1e-4):
    lh = lower_bound
    rh = upper_bound
    while abs(rh - lh) > tolerance:
        mh = np.mean([lh, rh])
        lx, mx, rx = (function(h) for h in (lh, mh, rh))
        if lx == target:
            return lh
        if rx == target:
            return rh

        if lx <= target and rx >= target:
            if mx > target:
                rh = mh
            else:
                lh = mh
        elif lx > target and rx < target:
            lh, rh = rh, lh
        else:
            return None
    return mh
