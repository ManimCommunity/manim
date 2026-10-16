"""Shortcut plays evaluate without drawing, reading back, or presenting frames."""

from unittest.mock import Mock

import numpy as np
import pytest

from manim import (
    Scene,
    Square,
    tempconfig,
)

RASTER_CALLS = ("render", "get_frame", "_prepare_animation", "_present_frozen_frame")


@pytest.fixture(params=["cairo", "opengl"])
def settings(request, tmp_path):
    with tempconfig(
        {
            "renderer": request.param,
            "format": "mp4",
            "frame_rate": 4,
            "pixel_width": 32,
            "pixel_height": 16,
            "live_preview": False,
            "disable_caching": True,
            "progress_bar": "none",
            "media_dir": str(tmp_path),
        }
    ):
        yield tmp_path


def spy_on_raster(monkeypatch, renderer):
    """Count every call that draws, reads back, or presents a frame."""
    counters = {}
    for name in RASTER_CALLS:
        original = getattr(renderer, name)
        counter = Mock(side_effect=original)
        counters[name] = counter
        monkeypatch.setattr(renderer, name, counter)
    return counters


def total_calls(counters):
    return sum(counter.call_count for counter in counters.values())


class ShortcutScene(Scene):
    def construct(self):
        self.add(Square(fill_opacity=1))
        self.play(Square().animate.shift(np.array([1.0, 0.0, 0.0])), run_time=0.5)
        self.wait(0.5, frozen_frame=True)
        start = self.time
        self.wait(1, stop_condition=lambda: self.time >= start + 0.5)


def test_excluded_plays_do_no_raster_work(settings, monkeypatch):
    with tempconfig({"from_animation_number": 10}):
        scene = ShortcutScene()
        counters = spy_on_raster(monkeypatch, scene.renderer)
        scene.render()

    assert scene.time == 1.5
    assert total_calls(counters) == 0


def test_cached_plays_do_no_raster_work(settings, monkeypatch):
    # Only stepped plays here: a frozen wait does not reuse its segment reliably on
    # OpenGL, which predates this change and would make the assertion backend-specific.
    class Stepped(Scene):
        def construct(self):
            square = Square(fill_opacity=1)
            self.add(square)
            self.play(square.animate.shift(np.array([1.0, 0.0, 0.0])), run_time=0.5)
            self.play(square.animate.shift(np.array([1.0, 0.0, 0.0])), run_time=0.5)

    with tempconfig({"disable_caching": False}):
        cold = Stepped()
        cold.render()

        warm = Stepped()
        counters = spy_on_raster(monkeypatch, warm.renderer)
        warm.render()

    assert warm.renderer.animations_hashes == cold.renderer.animations_hashes
    assert warm.time == cold.time
    assert total_calls(counters) == 0


def test_rendered_plays_in_a_partly_skipped_scene_still_draw(settings, monkeypatch):
    scene = ShortcutScene()
    counters = spy_on_raster(monkeypatch, scene.renderer)
    with tempconfig({"from_animation_number": 1}):
        scene.render()

    assert counters["_prepare_animation"].call_count == 2
    assert counters["_present_frozen_frame"].call_count == 1
    assert counters["render"].call_count > 0


def test_static_layer_is_not_stale_after_an_excluded_play(settings):
    """Skipping _prepare_animation must not leak a stale Cairo static image."""

    class Static(Scene):
        def construct(self):
            self.add(Square(fill_opacity=1).shift(np.array([-2.0, 0.0, 0.0])))
            mover = Square(fill_opacity=1)
            self.add(mover)
            self.play(mover.animate.shift(np.array([1.0, 0.0, 0.0])), run_time=0.5)
            self.play(mover.animate.shift(np.array([1.0, 0.0, 0.0])), run_time=0.5)
            self.captured = self.get_image()

    full = Static()
    full.render()
    partial = Static()
    with tempconfig({"from_animation_number": 1}):
        partial.render()

    np.testing.assert_array_equal(
        np.asarray(partial.captured), np.asarray(full.captured)
    )
