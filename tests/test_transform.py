import numpy as np
import pytest

from manim.animation.transform import ApplyMatrix
from manim.constants import DEGREES
from manim.mobject.geometry import Square


def test_apply_matrix_about_point():
    square_1 = Square()
    square_2 = square_1.copy()

    # rotation matrix
    matrix = [[0, -1], [1, 0]]
    about_point = np.asarray((-1.0, 0, 0.0))

    ApplyMatrix(matrix, square_1, about_point=about_point)
    square_2.rotate(90 * DEGREES, about_point=about_point)

    np.testing.assert_array_equal(square_1.points.all() == square_2.points.all())
