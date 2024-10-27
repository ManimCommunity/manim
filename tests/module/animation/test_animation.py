from __future__ import annotations

import pytest

from manim import FadeIn, Scene


@pytest.mark.parametrize(
    "run_time",
    [0, -1],
)
def test_animation_forbidden_run_time(run_time):
    test_scene = Scene()
    with pytest.raises(
        ValueError, match="Please set the run_time to a positive number."
    ):
        test_scene.play(FadeIn(None, run_time=run_time))


def test_animation_run_time_shorter_than_frame_rate(manim_caplog, config):
    test_scene = Scene()
    test_scene.play(FadeIn(None, run_time=1 / (config.frame_rate + 1)))
    assert "too short for the current frame rate" in manim_caplog.text


@pytest.mark.parametrize("frozen_frame", [False, True])
def test_wait_run_time_shorter_than_frame_rate(manim_caplog, frozen_frame):
    test_scene = Scene()
    test_scene.wait(1e-9, frozen_frame=frozen_frame)
    assert "too short for the current frame rate" in manim_caplog.text
