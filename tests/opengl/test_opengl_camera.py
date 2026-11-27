from __future__ import annotations

import numpy as np

from manim import DEGREES, RIGHT, Scene
from manim.renderer.opengl import OpenGLCamera
from manim.utils import opengl
from manim.utils.space_ops import rotation_about_z, rotation_matrix


def test_inverse_rotation_matrix_preserves_opengl_euler_convention(
    using_opengl_renderer,
):
    theta, phi, gamma = np.array([-37, 62, 18]) * DEGREES
    camera = OpenGLCamera().set_euler_angles(theta, phi, gamma)
    expected = (
        rotation_about_z(theta) @ rotation_matrix(phi, RIGHT) @ rotation_about_z(gamma)
    ).T
    np.testing.assert_allclose(camera.inverse_rotation_matrix, expected, atol=1e-12)


def test_horizontal_orbit_drag_preserves_camera_target(using_opengl_renderer):
    scene = Scene()
    camera = scene.camera
    assert isinstance(camera, OpenGLCamera)
    camera.set_position(np.array([3.0, 2.0, 1.0]))
    target = scene.camera_target = np.array([1.0, 1.0, 0.0])
    yaw = 0.4
    expected = (
        opengl.translation_matrix(*target)
        @ opengl.z_rotation_matrix(-yaw)
        @ opengl.translation_matrix(*-target)
        @ camera.model_matrix
    )

    scene.mouse_drag_orbit_controls(np.zeros(3), np.array([yaw, 0.0, 0.0]), 1, 0)

    np.testing.assert_allclose(camera.model_matrix, expected, atol=1e-12)


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
