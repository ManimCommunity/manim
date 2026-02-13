from __future__ import annotations

import numpy as np

from manim.constants import RIGHT
from manim.mobject.geometry.polygram import Square


def test_bounding_box(using_opengl_renderer):
    square = Square().move_to(RIGHT)
    np.testing.assert_array_equal(
        square.get_bounding_box(),
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
    )

    # Test that the bounding box can be indexed and set
    square.bounding_box[1] = 300

    np.testing.assert_array_equal(
        square.bounding_box,
        np.array([[0.0, -1.0, 0.0], [300.0, 300.0, 300.0], [2.0, 1.0, 0.0]]),
    )
