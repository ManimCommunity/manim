import numpy as np
import pytest

from manim.constants import OUT, RIGHT, UP
from manim.utils import opengl
from manim.utils.space_ops import rotation_matrix


def test_euler_rotation_defaults_to_identity():
    np.testing.assert_array_equal(opengl.rotation_matrix(), np.eye(4))


@pytest.mark.parametrize(
    ("x", "y", "z"),
    [(0.6, 0, 0), (0, -1.1, 0), (0, 0, 0.2), (0.6, -1.1, 0.2)],
)
def test_euler_rotation_preserves_xyz_order(x, y, z):
    expected = (
        opengl.x_rotation_matrix(x)
        @ opengl.y_rotation_matrix(y)
        @ opengl.z_rotation_matrix(z)
    )
    np.testing.assert_allclose(opengl.rotation_matrix(x, y, z), expected, atol=1e-12)


@pytest.mark.parametrize(
    ("builder", "axis"),
    [
        (opengl.x_rotation_matrix, RIGHT),
        (opengl.y_rotation_matrix, UP),
        (opengl.z_rotation_matrix, OUT),
    ],
)
def test_axis_rotation_matches_space_ops(builder, axis):
    np.testing.assert_allclose(
        builder(0.8), rotation_matrix(0.8, axis, homogeneous=True), atol=1e-12
    )
