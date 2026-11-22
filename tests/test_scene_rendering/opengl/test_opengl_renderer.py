from __future__ import annotations

import platform
from unittest.mock import Mock

import numpy as np
import pytest

from manim import RIGHT, Scene, Square
from manim.renderer.opengl import OpenGLRenderer
from tests.assert_utils import assert_file_exists
from tests.test_scene_rendering.simple_scenes import *


def test_file_output_disables_window(
    config, using_temp_opengl_config, disabling_caching
):
    """File output should disable the window by default."""
    scene = SquareToCircle()
    renderer = scene.renderer
    scene.render()
    assert renderer.window is None
    assert_file_exists(scene._get_manager().file_writer.final_file_path)


@pytest.mark.skip(reason="Temporarily skip due to failing in Windows CI")
def test_live_preview_opengl_render_with_movies(
    config,
    using_temp_opengl_config,
    live_preview_config_movie,
    disabling_caching,
):
    """Live preview can be displayed while movie output is enabled."""
    scene = SquareToCircle()
    renderer = scene.renderer
    with scene._get_manager():
        scene.render()
        assert renderer.window is not None
        assert_file_exists(scene._get_manager().file_writer.final_file_path)
    assert renderer.window is None


@pytest.mark.skipif(
    platform.processor() == "aarch64", reason="Fails on Linux-ARM runners"
)
def test_live_preview_opengl_render_with_image_sequence(
    using_temp_opengl_config,
    live_preview_config_pngs,
    disabling_caching,
):
    """Live preview can be displayed while an image sequence is written."""
    scene = SquareToCircle()
    renderer = scene.renderer
    with scene._get_manager():
        scene.render()
        assert renderer.window is not None
    assert renderer.window is None


def test_get_frame_with_live_preview_disabled(config, using_opengl_renderer):
    """Get frame has the correct dimensions without a live preview."""
    config.live_preview = False

    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)
    assert not config.live_preview

    renderer = scene.renderer
    frame = renderer.render(*scene._get_manager()._frame()).pixels()

    # height and width are flipped
    assert renderer.get_pixel_shape()[0] == frame.shape[1]
    assert renderer.get_pixel_shape()[1] == frame.shape[0]
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous


@pytest.mark.slow
def test_frames_have_the_scene_size_with_a_live_preview(config, using_opengl_renderer):
    # The window can have any size; frames are drawn offscreen at the scene's.
    config.live_preview = True
    config.window_size = (100, 300)

    scene = SquareToCircle()
    renderer = scene.renderer
    preview = scene._get_manager().preview
    preview.show()
    try:
        assert renderer.window is not None
        frame = renderer.render(*scene._get_manager()._frame())
        pixels = frame.pixels()
        assert pixels.shape == (config.pixel_height, config.pixel_width, 4)
        preview.present(frame)
        scene.add(Square(fill_opacity=1))
        image = scene.get_image()
        assert image.size == (config.pixel_width, config.pixel_height)
        assert not np.array_equal(image, pixels)
        # The image is drawn apart: the framebuffer still holds the frame.
        np.testing.assert_array_equal(renderer.get_frame(), pixels)
    finally:
        renderer.close()


def test_render_without_frame_output_skips_gpu_readback(
    config,
    using_opengl_renderer,
):
    config.format = "none"
    config.live_preview = False
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.get_frame = Mock(wraps=renderer.get_frame)

    renderer.render(*scene._get_manager()._frame())

    renderer.get_frame.assert_not_called()


def test_window_positions_map_to_scene_points(config, using_opengl_renderer):
    config.live_preview = True
    scene = SquareToCircle()
    scene.camera.frame.shift(RIGHT)
    preview = scene._get_manager().preview
    width, height = scene.camera.frame_width, scene.camera.frame_height
    center = scene.camera.frame_center

    np.testing.assert_allclose(preview._scene_point(0.5, 0.5, False), center)
    np.testing.assert_allclose(
        preview._scene_point(0, 1, False), center + [-width / 2, height / 2, 0]
    )
    np.testing.assert_allclose(preview._scene_point(0.25, 0, True), [width / 4, 0, 0])
    scene.renderer.close()


def test_scene_images_are_drawn_with_the_scene_settings(config, using_opengl_renderer):
    # Without an open context, images are drawn by an offscreen renderer that
    # takes the scene renderer's size and the camera's background, and no window.
    config.live_preview = False
    config.pixel_width, config.pixel_height = 64, 36
    scene = SquareToCircle()
    scene.camera.background_color = "#FF0000"
    image = scene.get_image()
    assert image.size == (64, 36)
    assert image.getpixel((0, 0))[:3] == (255, 0, 0)
    assert scene.renderer._context is None


def test_moving_camera_frame_animates(config, using_opengl_renderer):
    config.dry_run = True

    class PanAndZoom(Scene):
        def construct(self):
            self.add(Square())
            self.play(self.camera.frame.animate.scale(0.5).shift(RIGHT))

    scene = PanAndZoom()
    scene.render()
    np.testing.assert_allclose(scene.camera.frame_center, RIGHT, atol=1e-12)
