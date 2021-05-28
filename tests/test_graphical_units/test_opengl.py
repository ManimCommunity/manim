import pytest

from manim import *
from manim.opengl import *
from tests.test_graphical_units import test_geometry

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


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


modules_to_test = [test_geometry]

for module in modules_to_test:
    name = module.MODULE_NAME
    scenes = get_scenes_to_test(module.__name__)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        "scene_to_test",
        scenes,
        indirect=False,
        ids=map(lambda s: s[1].__name__, scenes),
    )
    def test_opengl_scene(scene_to_test: Scene, tmpdir, show_diff, opengl):
        class wrapped_scene(scene_to_test[1]):
            def construct(self):
                super().construct()
                self.wait()

        wrapped_scene.__name__ = scene_to_test[1].__name__

        GraphicalUnitTester(
            wrapped_scene,
            name,
            tmpdir,
            rgb_atol=50,
            opengl_enabled=opengl,
            opengl_test=True,
        ).test(show_diff=show_diff)
