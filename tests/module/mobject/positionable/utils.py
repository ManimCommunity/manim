import math
from itertools import product

import numpy as np

from manim.constants import DEGREES, X_AXIS, Y_AXIS, Z_AXIS
from manim.mobject.abstract.positionable import Positionable
from manim.typing import Point3D_Array

ANCHOR_POINTS = np.array(list(product((-1.0, 0.0, 1.0), repeat=3)))
"""All 27 anchor points."""

MAIN_AXES = np.array([X_AXIS, Y_AXIS, Z_AXIS])
"""The 3 main axes."""

ALL_AXES = np.array(list(product((-1.0, 0.0, 1.0), repeat=3)))
"""All 27 primary axes."""


DIMENSIONS = [0, 1, 2]
"""All 3 dimensions."""


CUBE_VERTICES = np.array(list(product([-1.0, 1.0], repeat=3)))
"""The vertices of a cube of size 2.

Has the unique characteristic that the anchor point is identical to the anchor vector.
"""

TRIANGLE_VERTICES = np.array([(-1, 0, 0), (1, 0, 0), (0, math.sqrt(2), 0)])
"""The vertices of a triangle."""

TRIANGLE_ORIGIN = np.array([0, math.sqrt(2) / 2, 0])
"""The origin of the triangle."""

POSITIONS: Point3D_Array = np.array([(-3, -2, -1), (0, 0, 0), (1, 2, 3)])
"""Some positions."""

ANGLES: np.ndarray = np.array([-720, -360, -90, -33.3, 0, 33.3, 90, 360, 720]) * DEGREES
"""Some angles."""


FACTORS: np.ndarray = np.array([-1, 0, 1, 2, 5], dtype=float)
"""Some factors."""

SIZES = [-2, -1, 0, 1, 2, 5]
"""Some sizes."""


class PositionableWithFamily(Positionable):
    def __init__(self, submobjects: list[Positionable]) -> None:
        super().__init__()
        self.submobjects = submobjects

    def get_family(self) -> list[Positionable]:
        return [self, *(mob for sub in self.submobjects for mob in sub.get_family())]
