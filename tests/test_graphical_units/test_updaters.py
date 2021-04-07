import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class UpdaterTest(Scene):
    def construct(self):
        dot = Dot()
        square = Square()
        self.add(dot, square)
        square.add_updater(lambda m: m.next_to(dot, RIGHT, buff=SMALL_BUFF))
        self.add(square)
        self.play(dot.animate.shift(UP * 2))
        square.clear_updaters()


class ValueTrackerTest(Scene):
    def construct(self):
        theta = ValueTracker(PI / 2)
        line = Line(ORIGIN, RIGHT)
        line.rotate(theta.get_value(), about_point=ORIGIN)
        self.add(line)
        self.wait()


class UpdateSceneDuringAnimationTest(Scene):
    def construct(self):
        def f(mob):
            self.add(Square())

        s = Circle().add_updater(f)
        self.play(Create(s))


class LastFrameWhenClearedTest(Scene):
    def construct(self):
        dot = Dot()
        square = Square()
        square.add_updater(lambda m: m.move_to(dot, UL))
        self.add(square)
        self.play(dot.animate.shift(UP * 2), rate_func=linear)
        square.clear_updaters()
        self.wait()


MODULE_NAME = "updaters"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
