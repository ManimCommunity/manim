import pytest

import manim as mn
from manim import color as C
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester


class LastFrame(mn.Scene):
    def construct(self):
        tick_start = 1.0
        tick_end = 3.0
        val_tracker = mn.ValueTracker(tick_start)
        square = mn.Square(fill_opacity=1).set_stroke(width=0)
        self.add(square)
        num_colors = 1000
        cols = mn.color_gradient([C.RED, C.WHITE, C.BLUE], num_colors)

        def col_uptater(mob):
            integ = int(
                (val_tracker.get_value() - tick_start)
                / (tick_end - tick_start)
                * (num_colors - 1)
            )
            mob.set_color(cols[integ])

        square.add_updater(col_uptater)
        self.play(val_tracker.set_value, tick_end, rate_func=mn.linear, run_time=3)


MODULE_NAME = "last_frame"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
