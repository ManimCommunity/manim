# to support ValuedPoint type inside ValuedPoint
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Point = np.ndarray
Func = Callable[[Point], float]


# TODO: actual x, y tolerance
TOL = 0.002


@dataclass
class ValuedPoint:
    """A position associated with the corresponding function value"""

    pos: Point
    val: float = None

    def calc(self, fn: Func):
        self.val = fn(self.pos)
        return self

    def __repr__(self):
        return f"({self.pos[0]},{self.pos[1]}; {self.val})"

    @staticmethod
    def midpoint(p1: ValuedPoint, p2: ValuedPoint, fn: Func):
        mid = (p1.pos + p2.pos) / 2
        return ValuedPoint(mid, fn(mid))

    @staticmethod
    def intersectZero(p1: ValuedPoint, p2: ValuedPoint, fn: Func):
        """Find the point on line p1--p2 with value 0"""
        denom = p1.val - p2.val
        k1 = -p2.val / denom
        k2 = p1.val / denom
        pt = k1 * p1.pos + k2 * p2.pos
        return ValuedPoint(pt, fn(pt))


def binary_search_zero(p1: ValuedPoint, p2: ValuedPoint, fn: Func):
    """Returns a pair `(point, is_zero: bool)`

    Use is_zero to make sure it's not an asymptote like at x=0 on f(x,y) = 1/(xy) - 1"""
    if np.max(np.abs(p2.pos - p1.pos)) < TOL:
        # Binary search stop condition: too small to matter
        pt = ValuedPoint.intersectZero(p1, p2, fn)
        is_zero = pt.val == 0 or (
            np.sign(pt.val - p1.val) == np.sign(p2.val - pt.val)
            and np.abs(pt.val < TOL)
        )
        return pt, is_zero
    else:
        # binary search
        mid = ValuedPoint.midpoint(p1, p2, fn)
        if mid.val == 0:
            return mid, True
        # (Group 0 with negatives)
        elif (mid.val > 0) == (p1.val > 0):
            return binary_search_zero(mid, p2, fn)
        else:
            return binary_search_zero(p1, mid, fn)
