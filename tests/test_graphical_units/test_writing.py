import pytest

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test
from manim import *


class TextTest(Scene):
    def construct(self):
        t = Text("testing", font="Arial")
        self.play(Animation(t))


class TextMobjectTest(Scene):
    def construct(self):
        constants.TEX_TEMPLATE = TexTemplate()
        t = TextMobject("Hello world !")
        self.play(Animation(t))


class TexMobjectTest(Scene):
    def construct(self):
        constants.TEX_TEMPLATE = TexTemplate()
        t = TexMobject("\\sum_{n=1}^\\infty "
                       "\\frac{1}{n^2} = \\frac{\\pi^2}{6}")
        self.play(Animation(t))


MODULE_NAME = "writing"


@pytest.mark.slow
@pytest.mark.parametrize("scene_to_test",
                         get_scenes_to_test(__name__),
                         indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME,
                        tmpdir).test(show_diff=show_diff)
