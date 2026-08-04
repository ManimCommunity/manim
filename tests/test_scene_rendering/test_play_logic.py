from __future__ import annotations

import sys
from unittest.mock import Mock

import pytest

from manim import (
    Animation,
    Dot,
    Mobject,
    Scene,
    ValueTracker,
    Wait,
    np,
)

from .simple_scenes import (
    SceneForFrozenFrameTests,
    SceneWithMultipleCalls,
    SceneWithNonStaticWait,
    SceneWithSceneUpdater,
    SceneWithStaticWait,
    SquareToCircle,
)


@pytest.mark.parametrize("frame_rate", argvalues=[15, 30, 60])
def test_t_values(config, using_temp_config, disabling_caching, frame_rate):
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
def test_t_values_with_skip_animations(using_temp_config, disabling_caching):
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


def test_static_wait_detection(using_temp_config, disabling_caching):
    """Test if a static wait (wait that freeze the frame) is correctly detected"""
    scene = SceneWithStaticWait()
    scene.render()
    # Test is is_static_wait of the Wait animation has been set to True by compile_animation_ata
    assert scene.animations[0].is_static_wait
    assert scene.is_current_animation_frozen_frame()


def test_non_static_wait_detection(using_temp_config, disabling_caching):
    scene = SceneWithNonStaticWait()
    scene.render()
    assert not scene.animations[0].is_static_wait
    assert not scene.is_current_animation_frozen_frame()
    scene = SceneWithSceneUpdater()
    scene.render()
    assert not scene.animations[0].is_static_wait
    assert not scene.is_current_animation_frozen_frame()


def test_wait_with_stop_condition(using_temp_config, disabling_caching):
    class TestScene(Scene):
        def construct(self):
            self.wait_until(lambda: self.time >= 1)
            assert self.time >= 1
            d = Dot()
            d.add_updater(lambda mobj, dt: self.add(Mobject()))
            self.add(d)
            self.play(Wait(run_time=5, stop_condition=lambda: len(self.mobjects) > 5))
            assert len(self.mobjects) > 5
            assert self.time < 2

    scene = TestScene()
    scene.render()


def test_frozen_frame(using_temp_config, disabling_caching):
    scene = SceneForFrozenFrameTests()
    scene.render()
    assert scene.mobject_update_count == 0
    assert scene.scene_update_count == 0


def test_t_values_with_cached_data(using_temp_config):
    """Test the proper generation and use of the t values when an animation is cached."""
    scene = SceneWithMultipleCalls()
    # Mocking the file_writer will skip all the writing process.
    scene.renderer.file_writer = Mock(scene.renderer.file_writer)
    scene.renderer.update_skipping_status = Mock()
    # Simulate that all animations are cached.
    scene.renderer.file_writer.is_already_cached.return_value = True
    scene.update_to_time = Mock()

    scene.render()
    assert scene.update_to_time.call_count == 10


def test_t_values_save_last_frame(config, using_temp_config):
    """Test that there is only one t value handled when only saving the last frame"""
    config.save_last_frame = True
    scene = SquareToCircle()
    scene.update_to_time = Mock()
    scene.render()
    scene.update_to_time.assert_called_once_with(1)


@pytest.mark.parametrize("frame_rate", argvalues=[15, 30, 60])
def test_dt_of_the_first_frame_of_an_animation(
    config, using_temp_config, disabling_caching, frame_rate
):
    """The first frame of an animation comes one frame period after the last
    frame of the animation before it, so its ``dt`` has to be that period.

    Regression test for #3005 and #4611: ``compile_animation_data`` reset
    ``last_t`` to 0, so the first frame of every ``play()`` and ``wait()`` was
    dispatched with ``dt=0``. Every dt-based updater held still for that frame,
    and the dt such an updater accumulated over an animation came out one frame
    short of its ``run_time`` -- an error that grew with the number of calls.
    """
    config.frame_rate = frame_rate

    class TestScene(Scene):
        def construct(self):
            self.dts: list[list[float]] = []
            dot = Dot()
            dot.add_updater(lambda mobj, dt: self.dts[-1].append(dt))
            self.add(dot)
            for _ in range(3):
                self.dts.append([])
                self.play(Animation(Mobject()), run_time=1)

    scene = TestScene()
    scene.render()

    frame_period = 1 / frame_rate
    for dts in scene.dts[1:]:
        # The trailing zero is the settling update at the end of play_internal,
        # which renders nothing. Every frame that is rendered advances the
        # clock by exactly one frame period...
        assert dts[-1] == 0
        np.testing.assert_allclose(dts[:-1], frame_period)
        assert len(dts[:-1]) == frame_rate
        # ...so an updater accumulates the run_time over the animation, and no
        # more. Before the fix this summed to run_time - 1 / frame_rate.
        np.testing.assert_allclose(sum(dts), 1)


def test_dt_of_the_very_first_frame_of_a_scene_is_zero(
    using_temp_config, disabling_caching
):
    """No frame precedes the first one, so no time has passed before it.

    This is the one exception to the rule pinned above, and it is deliberate:
    it keeps the opening frame of every scene exactly as it was.
    """

    class TestScene(Scene):
        def construct(self):
            self.dts: list[float] = []
            dot = Dot()
            dot.add_updater(lambda mobj, dt: self.dts.append(dt))
            self.add(dot)
            self.play(Animation(Mobject()), run_time=1)

    scene = TestScene()
    scene.render()
    assert scene.dts[0] == 0


def test_animate_with_changed_custom_attribute(using_temp_config):
    """Test that animating the change of a custom attribute
    using the animate syntax works correctly.
    """

    class CustomAnimateScene(Scene):
        def construct(self):
            vt = ValueTracker(0)
            vt.custom_attribute = "hello"
            self.play(vt.animate.set_value(42).set(custom_attribute="world"))
            assert vt.get_value() == 42
            assert vt.custom_attribute == "world"

    CustomAnimateScene().render()
