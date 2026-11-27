import numpy as np
import pytest

from manim import DEGREES, ORIGIN, UP, Scene, ThreeDScene


def test_mouse_controls_move_the_camera(dry_run):
    scene = ThreeDScene()
    camera = scene.camera
    scene.set_camera_orientation(phi=60 * DEGREES, theta=-30 * DEGREES)

    scene.mouse_drag_orbit_controls(ORIGIN, np.array([0.1, 0.2, 0]), 1, 0)
    assert camera.get_theta() == pytest.approx(-30 * DEGREES - 0.1)
    assert camera.get_phi() == pytest.approx(60 * DEGREES + 0.2)

    width = camera.frame_width
    scene.mouse_scroll_orbit_controls(ORIGIN, UP)
    assert camera.frame_width < width

    center = camera.frame_center
    scene.mouse_drag_orbit_controls(ORIGIN, np.array([0.1, 0, 0]), 4, 0)
    assert not np.allclose(camera.frame_center, center)


def test_left_drag_pans_a_flat_camera(dry_run):
    scene = Scene()
    scene.mouse_drag_orbit_controls(ORIGIN, np.array([0.1, 0, 0]), 1, 0)
    assert scene.camera.frame_center[0] < 0
