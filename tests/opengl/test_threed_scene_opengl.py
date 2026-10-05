from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from manim import DEGREES, ThreeDScene, Transform, config, linear


def test_set_camera_orientation(using_opengl_renderer):
    scene = ThreeDScene()
    scene.set_camera_orientation(
        phi=70 * DEGREES, theta=-35 * DEGREES, zoom=0.72, frame_center=[1, 2, 0]
    )
    theta, phi, _ = scene.camera.euler_angles
    assert phi == pytest.approx(70 * DEGREES)
    assert theta == pytest.approx(-35 * DEGREES)
    assert scene.camera.get_height() == pytest.approx(config.frame_height / 0.72)
    assert scene.camera.frame_center == pytest.approx([1, 2, 0])


def test_set_camera_orientation_zoom_is_absolute(using_opengl_renderer):
    scene = ThreeDScene()
    scene.set_camera_orientation(zoom=2)
    scene.set_camera_orientation(zoom=2)
    assert scene.camera.get_height() == pytest.approx(config.frame_height / 2)


def test_set_camera_orientation_focal_distance_warns(using_opengl_renderer):
    with pytest.warns(UserWarning, match="focal distance"):
        ThreeDScene().set_camera_orientation(focal_distance=5)


def _capture_camera_animation(scene, monkeypatch, **kwargs):
    # Exercise move_camera's real target construction without a render loop.
    play = Mock()
    monkeypatch.setattr(scene, "play", play)
    scene.move_camera(**kwargs, rate_func=linear)
    play.assert_called_once()
    (animation,), play_kwargs = play.call_args
    assert isinstance(animation, Transform)
    assert animation.mobject is scene.camera
    animation.rate_func = play_kwargs["rate_func"]
    return animation


def test_move_camera_interpolates_and_preserves_omitted_settings(
    using_opengl_renderer, monkeypatch
):
    scene = ThreeDScene()
    start_angles = np.array([20, 40, -10]) * DEGREES
    target_angles = np.array([-35, 65, 15]) * DEGREES
    start_center = np.array([-1, 0, 0.3])
    target_center = np.array([1, 0.5, 0.2])
    scene.set_camera_orientation(
        theta=start_angles[0],
        phi=start_angles[1],
        gamma=start_angles[2],
        zoom=0.72,
        frame_center=start_center,
    )
    start_height = scene.camera.get_height()
    target_zoom = 1.5
    target_height = config.frame_height / target_zoom
    aspect_ratio = scene.camera.get_width() / start_height
    animation = _capture_camera_animation(
        scene,
        monkeypatch,
        theta=target_angles[0],
        phi=target_angles[1],
        gamma=target_angles[2],
        zoom=target_zoom,
        frame_center=target_center,
    )
    animation.begin()

    for alpha in (0, 0.25, 0.5, 0.75, 1):
        animation.interpolate(alpha)
        expected_angles = (1 - alpha) * start_angles + alpha * target_angles
        expected_center = (1 - alpha) * start_center + alpha * target_center
        # Preserve OpenGL's frame-height interpolation, not linear zoom.
        expected_height = (1 - alpha) * start_height + alpha * target_height
        np.testing.assert_allclose(
            scene.camera.euler_angles, expected_angles, atol=1e-12
        )
        np.testing.assert_allclose(
            scene.camera.get_center(), expected_center, atol=1e-12
        )
        assert scene.camera.get_height() == pytest.approx(expected_height)
        assert scene.camera.get_width() == pytest.approx(aspect_ratio * expected_height)

    animation.finish()
    np.testing.assert_allclose(scene.camera.euler_angles, target_angles)
    np.testing.assert_allclose(scene.camera.get_center(), target_center)
    assert scene.camera.get_height() == pytest.approx(target_height)

    # Reuse the scene to check a partial update without another OpenGL context.
    partial_angles = target_angles.copy()
    partial_angles[1] += 15 * DEGREES
    animation = _capture_camera_animation(scene, monkeypatch, phi=partial_angles[1])
    animation.begin()
    for alpha in (0.5, 1):
        animation.interpolate(alpha)
        expected_angles = (1 - alpha) * target_angles + alpha * partial_angles
        np.testing.assert_allclose(scene.camera.euler_angles, expected_angles)
        np.testing.assert_allclose(scene.camera.get_center(), target_center)
        assert scene.camera.get_height() == pytest.approx(target_height)
        assert scene.camera.get_width() == pytest.approx(aspect_ratio * target_height)
    animation.finish()
    np.testing.assert_allclose(scene.camera.euler_angles, partial_angles)
