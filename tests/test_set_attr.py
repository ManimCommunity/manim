import numpy as np
import pytest

from manim.constants import RIGHT
from manim.opengl import OpenGLSquare


def test_Data():
    a = OpenGLSquare().move_to(RIGHT)
    data_bb = a.data["bounding_box"]
    assert np.array_equal(
        data_bb, np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    )

    # test that calling the attribute equals calling it from self.data
    assert np.array_equal(a.bounding_box, data_bb)

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

    data_bb = a.data["bounding_box"]
    assert np.array_equal(
        data_bb, np.array([[0.0, -1.0, 0.0], [300.0, 300.0, 300.0], [2.0, 1.0, 0.0]])
    )

    assert np.array_equal(a.bounding_box, data_bb)
