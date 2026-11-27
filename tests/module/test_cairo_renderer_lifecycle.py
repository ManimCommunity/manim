"""A Cairo renderer draws each frame with the settings the frame gives it."""

import numpy as np

from manim import RED, CairoRenderer, Camera, config, tempconfig
from manim.renderer.frame import FrameBuilder


def test_first_draw_uses_captured_settings_not_later_config():
    with tempconfig(
        {
            "pixel_width": 64,
            "pixel_height": 32,
            "background_color": RED,
        }
    ):
        renderer = CairoRenderer()
        camera = Camera()
        frames = FrameBuilder(config.pixel_width, config.pixel_height)
    try:
        with tempconfig({"pixel_width": 128, "pixel_height": 96}):
            frame = renderer.render(*frames.build(camera, [])).pixels()
        assert frame.shape == (32, 64, 4)
        assert np.all(frame == frame[0, 0])
        np.testing.assert_array_equal(frame[0, 0], RED.to_int_rgba())
    finally:
        renderer.close()


def test_one_renderer_draws_frames_of_several_sizes():
    # The target has the size of the frame it draws; a renderer has none of its own.
    from manim import Square

    with CairoRenderer() as renderer:
        for width, height in ((64, 36), (32, 48), (64, 36)):
            pixels = renderer.render(
                *FrameBuilder(width, height).build(Camera(), [Square(fill_opacity=1)])
            ).pixels()
            assert pixels.shape == (height, width, 4)
            assert pixels[height // 2, width // 2, :3].any()
