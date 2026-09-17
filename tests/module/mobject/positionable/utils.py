from itertools import product

import numpy as np

from manim.mobject.abstract.positionable import Positionable

ANCHOR_POINTS = np.array(list(product((-1.0, 0.0, 1.0), repeat=3)))
"""All 27 anchor points."""

AXES = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
"""All 27 possible axis."""


CUBE_VERTICES = np.array(list(product([-1.0, 1.0], repeat=3)))
"""The vertices of a cube of size 2.

Has the unique characteristic that the anchor point is identical to the anchor vector.
"""


class PositionableWithFamily(Positionable):
    def __init__(self, submobjects: list[Positionable]) -> None:
        super().__init__()
        self.submobjects = submobjects

    def get_family(self) -> list[Positionable]:
        return [self, *(mob for sub in self.submobjects for mob in sub.get_family())]
