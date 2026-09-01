from __future__ import annotations

import platform
from unittest.mock import Mock

import numpy as np
import pytest

from manim.renderer.opengl import OpenGLRenderer
from tests.assert_utils import assert_file_exists
from tests.test_scene_rendering.simple_scenes import *


def test_file_output_disables_window(
    config, using_temp_opengl_config, disabling_caching
):
    """File output should disable the window by default."""
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.update_frame = Mock(wraps=renderer.update_frame)
    scene.render()
    assert renderer.window is None
    assert_file_exists(renderer.file_writer.final_file_path)


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
    renderer.update_frame = Mock(wraps=renderer.update_frame)
    scene.render()
    assert renderer.window is not None
    assert_file_exists(renderer.file_writer.final_file_path)
    renderer.window.close()


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
    renderer.update_frame = Mock(wraps=renderer.update_frame)
    scene.render()
    assert renderer.window is not None
    renderer.window.close()


def test_get_frame_with_live_preview_disabled(config, using_opengl_renderer):
    """Get frame has the correct dimensions without a live preview."""
    config.live_preview = False

    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)
    assert not config.live_preview

    renderer = scene.renderer
    renderer.update_frame(scene)
    frame = renderer.get_frame()

    # height and width are flipped
    assert renderer.get_pixel_shape()[0] == frame.shape[1]
    assert renderer.get_pixel_shape()[1] == frame.shape[0]
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous


@pytest.mark.slow
def test_get_frame_with_live_preview_enabled(config, using_opengl_renderer):
    """Get frame has the correct dimensions with a live preview."""
    config.live_preview = True

    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)
    assert config.live_preview is True

    renderer = scene.renderer
    assert renderer.window is not None
    assert not renderer.file_writer.output_spec.enabled
    renderer.update_frame(scene)
    frame = renderer.get_frame()

    # height and width are flipped
    assert renderer.get_pixel_shape()[0] == frame.shape[1]
    assert renderer.get_pixel_shape()[1] == frame.shape[0]
    assert frame.dtype == np.uint8
    assert frame.flags.c_contiguous
    renderer.window.close()


def test_render_without_frame_output_skips_gpu_readback(
    config,
    using_opengl_renderer,
):
    config.format = "none"
    config.live_preview = False
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.get_frame = Mock(wraps=renderer.get_frame)

    renderer.render(scene, 0, [])

    renderer.get_frame.assert_not_called()


def test_pixel_coords_to_space_coords(config, using_opengl_renderer):
    config.live_preview = False

    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)

    renderer = scene.renderer
    renderer.update_frame(scene)

    px, py = 3, 2
    pw, ph = renderer.get_pixel_shape()
    _, fh = renderer.camera.get_shape()
    fc = renderer.camera.get_center()

    ex = fc[0] + (fh / ph) * (px - pw / 2)
    ey = fc[1] + (fh / ph) * (py - ph / 2)
    ez = fc[2]

    assert (
        renderer.pixel_coords_to_space_coords(px, py) == np.array([ex, ey, ez])
    ).all()
    assert (
        renderer.pixel_coords_to_space_coords(px, py, top_left=True)
        == np.array([ex, -ey, ez])
    ).all()
