"""Frame queries stay fresh without changing projection arithmetic."""

import numpy as np

from manim import (
    Camera,
    Mobject,
    ScreenRectangle,
    ThreeDCamera,
)


def test_the_frame_center_follows_the_frame():
    # The frame is an ordinary ScreenRectangle; its center is read fresh.
    camera = Camera()
    camera.frame.shift([2, -1, 0]).stretch(1.7, 0)
    np.testing.assert_allclose(camera.frame_center, [2, -1, 0])
    camera.frame.points[0] += [3, 4, 5]
    np.testing.assert_array_equal(camera.frame_center, Mobject.get_center(camera.frame))


def test_custom_frame_center_and_critical_point_hooks_remain_supported():
    class CustomFrame(ScreenRectangle):
        def get_critical_point(self, direction):
            return np.array([3.0, 4.0, 5.0])

    frame = CustomFrame()
    camera = ThreeDCamera(frame=frame)
    np.testing.assert_array_equal(camera.frame_center, [3, 4, 5])
    frame.get_center = lambda: np.array([6.0, 7.0, 8.0])
    np.testing.assert_array_equal(camera.frame_center, [6, 7, 8])
    assert camera.frame is frame
