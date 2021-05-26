import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class BraceSharpnessTest(Scene):
    def construct(self):
        line = Line(LEFT * 3, RIGHT * 3).shift(UP * 4)

        for sharpness in [0, 0.25, 0.5, 0.75, 1, 2, 3, 5]:
            self.add(Brace(line, sharpness=sharpness))
            line.shift(DOWN)
            self.wait()


class BraceTipTest(Scene):
    def construct(self):
        line = Line().shift(LEFT * 3).rotate(PI / 2)
        steps = 8
        for i in range(steps):
            brace = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
            dot = Dot()
            brace.put_at_tip(dot)
            line.rotate_about_origin(TAU / steps)
            self.add(brace, dot)
            self.wait()


class ArcBraceTest(Scene):
    def construct(self):
        self.play(Animation(ArcBrace()))


MODULE_NAME = "brace"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
