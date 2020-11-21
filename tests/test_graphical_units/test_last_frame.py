import pytest

import manim as mn
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester

mn.config.frame_rate = 5
mn.config.pixel_height = 150
mn.config.pixel_width = 150


def make_lbl(num):
    text = mn.Text(f"{num}").scale(7)
    text.num = num
    return text


def update_lbl(lbl, dt):
    lbl.num += 1
    lbl.become(make_lbl(lbl.num))


class LastFrame(mn.Scene):
    def construct(self):
        text = make_lbl(0).add_updater(update_lbl)
        self.add(text)
        sq = mn.Square(stroke_width=30).scale(3)
        self.play(mn.ShowCreation(sq), run_time=1)


MODULE_NAME = "last_frame"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
