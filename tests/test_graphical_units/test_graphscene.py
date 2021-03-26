import pytest

from manim import *
from ..utils.testing_utils import get_scenes_to_test
from ..utils.GraphicalUnitTester import GraphicalUnitTester


class PlotFunctions(Scene):
    def construct(self):
        graph = Axes(
            x_range=[-10, 10.3],
            y_range=[-1.5, 1.5],
            x_length=9,
            y_length=6,
            axis_config={"color": GREEN, "include_tip": False, "stroke_width": 4},
            x_axis_config={
                "numbers_to_exclude": [x for x in range(-9, 11, 2)] + [0],
                "include_numbers": True,
                "numbers_with_elongated_ticks": range(-10, 12, 2),
            },
        )
        labels = graph.get_axis_labels()
        constants.TEX_TEMPLATE = TexTemplate()
        f = graph.get_graph(lambda x: x ** 2, color=BLUE)
        self.play(Animation(graph), Animation(labels), Animation(f))


MODULE_NAME = "plot"


@pytest.mark.slow
@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
