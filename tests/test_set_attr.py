from __future__ import annotations

import numpy as np
import pytest

from manim import config
from manim.constants import RIGHT
from manim.mobject.geometry import Square


def test_Data():
    config.renderer = "opengl"
    a = Square().move_to(RIGHT)
    bb = a.bounding_box
    assert np.array_equal(
        bb,
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]),
    )

    # test that calling the attribute equals calling it from self.data
    assert np.array_equal(a.bounding_box, bb)

    # test that the array can be indexed
    assert np.array_equal(
        a.bounding_box[1],
        np.array(
            [1.0, 0.0, 0.0],
        ),
    )

    # test that a value can be set
    a.bounding_box[1] = 300

    # test that both the attr and self.data arrays match after adjusting a value

    bb = a.bounding_box
    assert np.array_equal(
        bb,
        np.array([[0.0, -1.0, 0.0], [300.0, 300.0, 300.0], [2.0, 1.0, 0.0]]),
    )

    assert np.array_equal(a.bounding_box, bb)
    config.renderer = "cairo"  # needs to be here or else the following cairo tests fail
