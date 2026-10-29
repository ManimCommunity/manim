"""A Cairo renderer draws with the settings it was created with."""

import numpy as np

from manim import RED, CairoRenderer, tempconfig


def test_first_draw_uses_captured_settings_not_later_config():
    with tempconfig(
        {
            "pixel_width": 64,
            "pixel_height": 32,
            "background_color": RED,
        }
    ):
        renderer = CairoRenderer()
    try:
        with tempconfig({"pixel_width": 128, "pixel_height": 96}):
            renderer.render_mobjects([])
        frame = renderer.get_frame()
        assert frame.shape == (32, 64, 4)
        assert np.all(frame == frame[0, 0])
        np.testing.assert_array_equal(frame[0, 0], RED.to_int_rgba())
    finally:
        renderer.close()
