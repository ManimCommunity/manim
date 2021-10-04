from unittest.mock import Mock

import pytest

from manim import *
from manim import config

from ..simple_scenes import (
    SceneWithMultipleCalls,
    SceneWithNonStaticWait,
    SceneWithStaticWait,
    SquareToCircle,
)


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Mock object has a different implementation in python 3.7, which makes it broken with this logic.",
)
@pytest.mark.parametrize("frame_rate", argvalues=[15, 30, 60])
def test_t_values(using_temp_opengl_config, disabling_caching, frame_rate):
    """Test that the framerate corresponds to the number of t values generated"""
    config.frame_rate = frame_rate
    scene = SquareToCircle()
    scene.update_to_time = Mock()
    scene.render()
    assert scene.update_to_time.call_count == config["frame_rate"]
    np.testing.assert_allclose(
        ([call.args[0] for call in scene.update_to_time.call_args_list]),
        np.arange(0, 1, 1 / config["frame_rate"]),
    )


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Mock object has a different implementation in python 3.7, which makes it broken with this logic.",
)
def test_t_values_with_skip_animations(using_temp_opengl_config, disabling_caching):
    """Test the behaviour of scene.skip_animations"""
    scene = SquareToCircle()
    scene.update_to_time = Mock()
    scene.renderer._original_skipping_status = True
    scene.render()
    assert scene.update_to_time.call_count == 1
    np.testing.assert_almost_equal(
        scene.update_to_time.call_args.args[0],
        1.0,
    )


@pytest.mark.xfail(reason="Not currently implemented for opengl")
def test_static_wait_detection(using_temp_opengl_config, disabling_caching):
    """Test if a static wait (wait that freeze the frame) is correctly detected"""
    scene = SceneWithStaticWait()
    scene.render()
    # Test is is_static_wait of the Wait animation has been set to True by compile_animation_ata
    assert scene.animations[0].is_static_wait
    assert scene.is_current_animation_frozen_frame()


def test_non_static_wait_detection(using_temp_opengl_config, disabling_caching):
    scene = SceneWithNonStaticWait()
    scene.render()
    assert not scene.animations[0].is_static_wait
    assert not scene.is_current_animation_frozen_frame()


@pytest.mark.xfail(reason="Should be fixed in #2133")
def test_t_values_with_cached_data(using_temp_opengl_config):
    """Test the proper generation and use of the t values when an animation is cached."""
    scene = SceneWithMultipleCalls()
    # Mocking the file_writer will skip all the writing process.
    scene.renderer.file_writer = Mock(scene.renderer.file_writer)
    # Simulate that all animations are cached.
    scene.renderer.file_writer.is_already_cached.return_value = True
    scene.update_to_time = Mock()

    scene.render()
    assert scene.update_to_time.call_count == 10


@pytest.mark.xfail(reason="Not currently handled correctly for opengl")
def test_t_values_save_last_frame(using_temp_opengl_config):
    """Test that there is only one t value handled when only saving the last frame"""
    config.save_last_frame = True
    scene = SquareToCircle()
    scene.update_to_time = Mock()
    scene.render()
    scene.update_to_time.assert_called_once_with(1)
