import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class FocusOnTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(FocusOn(square))


class IndicateTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(Indicate(square))


class FlashTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(Flash(ORIGIN))


class CircumscribeTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(Circumscribe(square))
        self.wait()


class ShowPassingFlashTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(ShowPassingFlash(square.copy()))


class ShowCreationThenFadeOutTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(ShowCreationThenFadeOut(square.copy()))


class ApplyWaveTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(ApplyWave(square))


class WiggleTest(Scene):
    def construct(self):
        square = Square()
        self.add(square)
        self.play(Wiggle(square))


MODULE_NAME = "indication"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
