from __future__ import annotations

import numpy as np

from manim import config
from manim.constants import RIGHT
from manim.mobject.geometry.polygram import Square


def test_Data():
    config.renderer = "opengl"
    a = Square().move_to(RIGHT)
    data_bb = a.data["bounding_box"]
    np.testing.assert_array_equal(
        data_bb,
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
    )

    # test that calling the attribute equals calling it from self.data
    np.testing.assert_array_equal(a.bounding_box, data_bb)

    # test that the array can be indexed
    np.testing.assert_array_equal(
        a.bounding_box[1],
        np.array(
            [1.0, 0.0, 0.0],
        ),
    )

    # test that a value can be set
    a.bounding_box[1] = 300

    # test that both the attr and self.data arrays match after adjusting a value

    data_bb = a.data["bounding_box"]
    np.testing.assert_array_equal(
        data_bb,
        np.array([[0.0, -1.0, 0.0], [300.0, 300.0, 300.0], [2.0, 1.0, 0.0]]),
    )

    np.testing.assert_array_equal(a.bounding_box, data_bb)
    config.renderer = "cairo"  # needs to be here or else the following cairo tests fail
