"""Default frame queries stay fresh without changing projection arithmetic."""

import numpy as np
import pytest

from manim import (
    RIGHT,
    Camera,
    Mobject,
    ScreenRectangle,
    Square,
    ThreeDCamera,
    VMobject,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_default_frame_center_matches_generic_query_after_mutations(dtype):
    camera = Camera()
    frame = camera.frame
    frame.points = frame.points.astype(dtype)

    def check():
        np.testing.assert_array_equal(
            frame.get_points_defining_boundary(),
            VMobject.get_points_defining_boundary(frame),
        )
        np.testing.assert_array_equal(camera.frame_center, Mobject.get_center(frame))
        assert camera.frame_center.dtype == np.float64

    check()
    frame.shift([2, -1, 0.5]).rotate(0.3).stretch(1.7, 0)
    check()
    # Direct point edits must not require a draw or cache invalidation.
    frame.points[0] += [3, 4, 5]
    check()
    child = Square().shift(10 * RIGHT)
    frame.add(child)
    check()
    child.shift(5 * RIGHT)
    check()
    frame.remove(child)
    frame.clear_points()
    check()
    frame.set_points(np.array([[1.0, 2.0, 3.0]]))
    check()
    boundary = frame.get_points_defining_boundary()
    boundary[:] = 0
    np.testing.assert_array_equal(frame.points, [[1, 2, 3]])


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
