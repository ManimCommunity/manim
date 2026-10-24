from __future__ import annotations

import numpy as np
import pytest

from manim.utils.qhull import QuickHull


def test_initial_simplex_is_affinely_independent():
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],  # The first three points are collinear.
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )

    simplex = QuickHull()._find_initial_simplex(points)

    assert np.linalg.matrix_rank(simplex[1:] - simplex[0]) == points.shape[1]


def test_quickhull_rejects_degenerate_input():
    collinear_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )

    with pytest.raises(ValueError, match="full coordinate dimension"):
        QuickHull().build(collinear_points)
