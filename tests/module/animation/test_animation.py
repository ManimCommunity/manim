from __future__ import annotations

import pytest

from manim import FadeIn


def test_empty_animation_fails():
    with pytest.raises(ValueError, match="Please set the run_time to be positive"):
        FadeIn(None, run_time=0).begin()


def test_negative_run_time_fails():
    with pytest.raises(ValueError, match="Please set the run_time to be positive"):
        FadeIn(None, run_time=-1).begin()


def test_run_time_shorter_than_frame_rate(caplog):
    FadeIn(None, run_time=1e-9).begin()
    assert (
        "Original run time of FadeIn(Mobject) is shorter than current frame rate"
        in caplog.text
    )
