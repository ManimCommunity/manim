"""Every play steps through all of its frames; listeners decide what is drawn."""

from unittest.mock import patch

import numpy as np
import pytest

from manim import LEFT, RIGHT, Dot, Manager, Scene, Square, tempconfig
from manim.renderer.cairo.renderer import CairoRenderer

SETTINGS = {
    "renderer": "cairo",
    "format": "none",
    "frame_rate": 4,
    "pixel_width": 32,
    "pixel_height": 16,
    "disable_caching": True,
    "progress_bar": "none",
}


class Stateful(Scene):
    """An updater whose result depends on how often it runs."""

    def construct(self):
        self.calls = 0

        def count(mobject, dt):
            self.calls += 1
            mobject.shift(dt * RIGHT)

        self.dot = Dot().add_updater(count)
        self.add(self.dot)
        self.next_section("intro", skip_animations=self.skip_intro)
        self.play(Square().animate.shift([0, 1, 0]), run_time=1)
        self.wait(0.5, frozen_frame=False)
        self.next_section("main")
        self.play(Square().animate.shift([0, -1, 0]), run_time=1)


@pytest.mark.parametrize(
    "skip",
    [
        {"section": True},
        {"config": {"from_animation_number": 2}},
        {"config": {"save_last_frame": True}},
    ],
    ids=["skipped section", "-n", "-s"],
)
def test_skipped_plays_reach_the_state_of_a_full_render(skip, tmp_path):
    settings = {**SETTINGS, "media_dir": str(tmp_path)}
    with tempconfig(settings):
        full = Stateful()
        full.skip_intro = False
        Manager(full).render()
    with tempconfig({**settings, **skip.get("config", {})}):
        skipped = Stateful()
        skipped.skip_intro = skip.get("section", False)
        Manager(skipped).render()
    assert skipped.calls == full.calls
    assert skipped.dot.get_center() == pytest.approx(full.dot.get_center())
    assert skipped.time == full.time


def test_a_frozen_wait_shows_no_progress_bar(tmp_path):
    class Frozen(Scene):
        def construct(self):
            self.wait(1, frozen_frame=True)

    with tempconfig({**SETTINGS, "media_dir": str(tmp_path)}):
        scene = Frozen()
        progression = patch.object(scene, "_get_animation_time_progression")
        with progression as bar:
            Manager(scene).render()
    bar.assert_not_called()


def test_the_final_state_is_drawn_after_a_skipped_render(tmp_path):
    class Moves(Scene):
        def construct(self):
            self.add(Square(fill_opacity=1).shift(3 * RIGHT))
            self.play(Square(fill_opacity=1).animate.shift(3 * LEFT), run_time=0.5)

    with tempconfig({**SETTINGS, "media_dir": str(tmp_path)}):
        drawn = Moves(skip_animations=True)
        with drawn._get_manager():
            drawn.render()
            image = drawn.renderer.get_frame().copy()
        expected = Moves()
        with expected._get_manager():
            expected.render()
            final = np.asarray(expected.get_image())
    np.testing.assert_array_equal(image, final)


def test_each_frame_is_drawn_once_for_all_listeners(tmp_path):
    with tempconfig({**SETTINGS, "media_dir": str(tmp_path)}):
        scene = Stateful()
        scene.skip_intro = False
        with patch.object(
            CairoRenderer, "render", autospec=True, side_effect=CairoRenderer.render
        ) as render:
            Manager(scene).render()
    # 1 s + 0.5 s + 1 s at 4 fps; the encoder and the preview share each drawing.
    assert render.call_count == 4 + 2 + 4
