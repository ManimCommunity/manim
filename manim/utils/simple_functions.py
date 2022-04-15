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
from types import MappingProxyType
from typing import Callable, Union

import numpy as np
from scipy import special


def sigmoid(x: float) -> float:
    return 1.0 / (1 + np.exp(-x))


@lru_cache(maxsize=10)
def choose(n: int, k: int) -> int:
    return special.comb(n, k, exact=True)


def get_parameters(function: Callable) -> MappingProxyType[str, inspect.Parameter]:
    return inspect.signature(function).parameters


def clip(a, min_a, max_a):
    """Any comparable objects (i.e. supports >,<)"""
    if a < min_a:
        return min_a
    elif a > max_a:
        return max_a
    return a


def binary_search(
    function: Callable[[Union[int, float]], Union[int, float]],
    target: Union[int, float],
    lower_bound: Union[int, float],
    upper_bound: Union[int, float],
    tolerance: Union[int, float] = 1e-4,
) -> Union[int, float, None]:
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
