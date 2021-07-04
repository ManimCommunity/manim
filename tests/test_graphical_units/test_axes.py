import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test


class AxesTest(Scene):
    def construct(self):
        graph = Axes(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            x_length=6,
            y_length=6,
            color=WHITE,
            axis_config={"exclude_origin_tick": False},
        )
        labels = graph.get_axis_labels()
        self.add(graph, labels)


MODULE_NAME = "plot"


@pytest.mark.slow
@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
