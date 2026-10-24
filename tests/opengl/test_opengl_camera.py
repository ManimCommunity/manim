from __future__ import annotations

import numpy as np

from manim import DEGREES
from manim.renderer.opengl_renderer import OpenGLCamera


def test_camera_interpolate_angles(using_opengl_renderer):
    # Literal -340-degree theta change and two positive gamma revolutions.
    start_angles = np.array([350, 40, -10]) * DEGREES
    target_angles = np.array([10, 65, 710]) * DEGREES
    start = OpenGLCamera().set_euler_angles(*start_angles)
    target = start.copy().set_euler_angles(*target_angles)
    camera = start.copy()
    reference = start.copy()
    start_matrix = start.inverse_rotation_matrix.copy()
    target_matrix = target.inverse_rotation_matrix.copy()

    # Repeated and out-of-order samples must still use the fixed endpoints.
    for alpha in (0, 0.25, 0.75, 0.5, 0.5, 1):
        expected = (1 - alpha) * start_angles + alpha * target_angles
        assert camera.interpolate(start, target, alpha) is camera
        np.testing.assert_allclose(camera.euler_angles, expected, atol=1e-12)
        reference.set_euler_angles(*expected)
        np.testing.assert_allclose(
            camera.inverse_rotation_matrix,
            reference.inverse_rotation_matrix,
            atol=1e-12,
        )
        np.testing.assert_array_equal(start.euler_angles, start_angles)
        np.testing.assert_array_equal(target.euler_angles, target_angles)
        np.testing.assert_array_equal(start.inverse_rotation_matrix, start_matrix)
        np.testing.assert_array_equal(target.inverse_rotation_matrix, target_matrix)


def test_camera_interpolate_keeps_point_path_separate(using_opengl_renderer):
    start = OpenGLCamera().set_euler_angles(0.2, 0.4, -0.1)
    target = start.copy().set_euler_angles(0.7, 0.9, 0.3).move_to([1, 2, 0.5])
    camera = start.copy()

    def curved_path(start_points, target_points, alpha):
        return (
            (1 - alpha) * start_points
            + alpha * target_points
            + np.array([0, 0, np.sin(np.pi * alpha)])
        )

    camera.interpolate(start, target, 0.5, path_func=curved_path)
    np.testing.assert_allclose(
        camera.points, (start.points + target.points) / 2 + [0, 0, 1]
    )
    np.testing.assert_allclose(
        camera.euler_angles, (start.euler_angles + target.euler_angles) / 2
    )


def test_camera_animate_interpolates_before_finish(using_opengl_renderer):
    start_angles = np.array([20, 40, -10]) * DEGREES
    target_angles = start_angles.copy()
    target_angles[0] -= 360 * DEGREES
    camera = OpenGLCamera().set_euler_angles(*start_angles)
    reference = camera.copy()
    animation = (
        camera.animate(rate_func=lambda t: t * t).set_theta(target_angles[0]).build()
    )
    animation.begin()

    for alpha in (0, 0.25, 0.5, 0.75, 1):
        progress = alpha * alpha
        expected = (1 - progress) * start_angles + progress * target_angles
        animation.interpolate(alpha)
        np.testing.assert_allclose(camera.euler_angles, expected, atol=1e-12)
        reference.set_euler_angles(*expected)
        np.testing.assert_allclose(
            camera.inverse_rotation_matrix,
            reference.inverse_rotation_matrix,
            atol=1e-12,
        )

    # The alpha=1 assertion above must pass before finish replays the setter.
    animation.finish()
    np.testing.assert_allclose(camera.euler_angles, target_angles)
