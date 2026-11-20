"""3D projection follows the camera's current angles, drawn or not."""

import numpy as np
import pytest

from manim import ThreeDCamera


def test_projection_follows_the_angles_without_drawing():
    camera = ThreeDCamera()
    camera.set_theta(0)
    point = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(camera.project_point(point), [0, -1, 0], atol=1e-12)
    camera.set_theta(-90 * np.pi / 180)
    np.testing.assert_allclose(camera.project_point(point), [1, 0, 0], atol=1e-12)
    np.testing.assert_array_equal(
        camera.rotation_matrix, camera.generate_rotation_matrix()
    )


def test_rotation_matrix_is_read_only():
    camera = ThreeDCamera()
    with pytest.raises(AttributeError):
        camera.rotation_matrix = np.identity(3)
