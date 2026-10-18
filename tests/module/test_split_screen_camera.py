from __future__ import annotations

import numpy as np
import pytest

from manim import BLUE, RED, Camera
from manim.camera.mapping_camera import SplitScreenCamera


@pytest.mark.parametrize("pixel_width", [20, 21])
def test_split_screen_camera_initializes_parent_once(monkeypatch, pixel_width):
    left = Camera(pixel_height=8, pixel_width=20, background_color=RED)
    right = Camera(pixel_height=8, pixel_width=20, background_color=BLUE)
    calls = []
    original_init = Camera.__init__

    def count_init(self, **kwargs):
        calls.append(self)
        original_init(self, **kwargs)

    monkeypatch.setattr(Camera, "__init__", count_init)
    camera = SplitScreenCamera(left, right, pixel_width=pixel_width, pixel_height=8)
    assert calls == [camera]
    assert camera.pixel_width == pixel_width
    assert camera.pixel_height == 8
    split = (pixel_width + 1) // 2
    assert left.pixel_width == split
    assert right.pixel_width == pixel_width - split
    camera.capture_mobjects([])
    np.testing.assert_array_equal(camera.pixel_array[:, :split], left.pixel_array)
    np.testing.assert_array_equal(camera.pixel_array[:, split:], right.pixel_array)
