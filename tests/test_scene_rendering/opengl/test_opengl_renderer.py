from __future__ import annotations

import platform
from unittest.mock import Mock

import numpy as np
import pytest

from manim.renderer.opengl_renderer import OpenGLRenderer
from tests.assert_utils import assert_file_exists
from tests.test_scene_rendering.simple_scenes import *


def test_write_to_movie_disables_window(
    config, using_temp_opengl_config, disabling_caching
):
    """write_to_movie should disable window by default"""
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.update_frame = Mock(wraps=renderer.update_frame)
    scene.render()
    assert renderer.window is None
    assert_file_exists(config.output_file)


@pytest.mark.skip(reason="Temporarily skip due to failing in Windows CI")
def test_force_window_opengl_render_with_movies(
    config,
    using_temp_opengl_config,
    force_window_config_write_to_movie,
    disabling_caching,
):
    """force_window creates window when write_to_movie is set"""
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.update_frame = Mock(wraps=renderer.update_frame)
    scene.render()
    assert renderer.window is not None
    assert_file_exists(config["output_file"])
    renderer.window.close()


@pytest.mark.skipif(
    platform.processor() == "aarch64", reason="Fails on Linux-ARM runners"
)
def test_force_window_opengl_render_with_format(
    using_temp_opengl_config,
    force_window_config_pngs,
    disabling_caching,
):
    """force_window creates window when format is set"""
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.update_frame = Mock(wraps=renderer.update_frame)
    scene.render()
    assert renderer.window is not None
    renderer.window.close()


def test_get_frame_with_preview_disabled(config, using_opengl_renderer):
    """Get frame is able to fetch frame with the correct dimensions when preview is disabled"""
    config.preview = False

    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)
    assert not config.preview

    renderer = scene.renderer
    renderer.update_frame(scene)
    frame = renderer.get_frame()

    # height and width are flipped
    assert renderer.get_pixel_shape()[0] == frame.shape[1]
    assert renderer.get_pixel_shape()[1] == frame.shape[0]


@pytest.mark.slow
def test_get_frame_with_preview_enabled(config, using_opengl_renderer):
    """Get frame is able to fetch frame with the correct dimensions when preview is enabled"""
    config.preview = True

    scene = SquareToCircle()
    assert isinstance(scene.renderer, OpenGLRenderer)
    assert config.preview is True

    renderer = scene.renderer
    renderer.update_frame(scene)
    frame = renderer.get_frame()

    # height and width are flipped
    assert renderer.get_pixel_shape()[0] == frame.shape[1]
    assert renderer.get_pixel_shape()[1] == frame.shape[0]


def test_pixel_coords_to_space_coords(config, using_opengl_renderer):
    config.preview = True

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
