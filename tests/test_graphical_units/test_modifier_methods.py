import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class GradientTest(Scene):
    def construct(self):
        c = Circle(fill_opacity=1).set_color(color=[YELLOW, GREEN])
        self.add(c)


class GradientRotationTest(Scene):
    def construct(self):
        c = Circle(fill_opacity=1).set_color(color=[YELLOW, GREEN]).rotate(PI)
        self.add(c)


MODULE_NAME = "modifier_methods"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
