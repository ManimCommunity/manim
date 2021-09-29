from unittest.mock import Mock

import pytest

from tests.assert_utils import assert_file_exists
from tests.test_scene_rendering.simple_scenes import *


def test_write_to_movie_disables_window(using_temp_opengl_config, disabling_caching):
    """write_to_movie should disable window by default"""
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.update_frame = Mock(wraps=renderer.update_frame)
    scene.render()
    assert renderer.window is None
    assert_file_exists(config["output_file"])


@pytest.mark.skip(msg="Temporarily skip due to failing in Windows CI")
def test_force_window_opengl_render_with_movies(
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
