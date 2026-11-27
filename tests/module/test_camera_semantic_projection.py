"""3D projection follows the camera's current angles, drawn or not."""

import numpy as np
import pytest

from manim import DEGREES, RIGHT, ThreeDCamera
from manim.utils.space_ops import rotation_about_z, rotation_matrix


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


def test_rotation_matrix_preserves_cairo_euler_convention():
    theta, phi, gamma = np.array([-37, 62, 18]) * DEGREES
    camera = ThreeDCamera(theta=theta, phi=phi, gamma=gamma)
    expected = (
        rotation_about_z(gamma)
        @ rotation_matrix(-phi, RIGHT)
        @ rotation_about_z(-theta - 90 * DEGREES)
    )
    np.testing.assert_allclose(camera.generate_rotation_matrix(), expected, atol=1e-12)


def test_rotation_matrix_is_read_only():
    camera = ThreeDCamera()
    with pytest.raises(AttributeError):
        camera.rotation_matrix = np.identity(3)


def test_points_are_projected_with_the_perspective_factor():
    camera = ThreeDCamera(focal_distance=10.0)
    camera.set_phi(0)
    camera.set_theta(-np.pi / 2)
    near = camera.project_point(np.array([1.0, 0.0, 5.0]))
    # x and y scale by focal_distance / (focal_distance - z).
    np.testing.assert_allclose(near[:2], [2.0, 0.0], atol=1e-12)
    with pytest.raises(TypeError, match="exponential_projection"):
        ThreeDCamera(exponential_projection=True)
