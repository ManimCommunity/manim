import os
from unittest.mock import Mock

from manim import *

from ..assert_utils import assert_file_exists
from .simple_scenes import *


def test_render(using_temp_config, disabling_caching):
    scene = SquareToCircle()
    renderer = scene.renderer
    renderer.update_frame = Mock()
    renderer.add_frame = Mock()
    scene.render()
    assert renderer.add_frame.call_count == config["frame_rate"]
    assert renderer.update_frame.call_count == config["frame_rate"]


def test_skipping_status_with_from_to_and_up_to(using_temp_config, disabling_caching):
    """Test if skip_animations is well udpated when -n flag is passed"""
    config.from_animation_number = 2
    config.upto_animation_number = 6

    class SceneWithMultipleCalls(Scene):
        def construct(self):
            number = Integer(0)
            self.add(number)
            for i in range(10):
                self.play(Animation(Square()))

                assert ((i >= 2) and (i <= 6)) or self.renderer.skip_animations

    SceneWithMultipleCalls().render()


def test_when_animation_is_cached(using_temp_config):
    scene = SquareToCircle()
    # Render twice to create a cache.
    scene.render()
    scene.render()
    assert scene.renderer.file_writer.is_already_cached(
        scene.renderer.animations_hashes[0]
    )
    assert_file_exists(config["output_file"])
