import pytest

from manim import *
from manim.opengl import *
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester

with tempconfig({"use_opengl_renderer": True}):

    class CircleTest(Scene):
        def construct(self):
            circle = OpenGLCircle().set_color(RED)
            self.add(circle)
            self.wait()


MODULE_NAME = "opengl"


@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    with tempconfig({"use_opengl_renderer": True}):
        # allow 1/255 RGB value differences with opengl tests because of differences across platforms
        GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir, rgb_atol=1.01).test(
            show_diff=show_diff
        )
