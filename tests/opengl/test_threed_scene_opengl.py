from __future__ import annotations

import pytest

from manim import DEGREES, ThreeDScene, config


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
