import pytest

from manim import *

from ..utils.GraphicalUnitTester import GraphicalUnitTester
from ..utils.testing_utils import get_scenes_to_test

EPSILON = 1e-5

class PlotFunctions(GraphScene):
    def __init__(self, **kwargs):
        GraphScene.__init__(
            self,
            graph_origin=ORIGIN,
            function_color=RED,
            axes_color=GREEN,
            x_labeled_nums=range(-10, 12, 2),
            **kwargs
        )

    def construct(self):
        self.x_min = -10
        self.x_max = 10.3
        self.y_min = -1.5
        self.y_max = 1.5

        constants.TEX_TEMPLATE = TexTemplate()
        self.setup_axes()
        f = self.get_graph(lambda x: x ** 2)

        self.add(f)

class PlotFunctionWithDiscontinuitiesTest(GraphScene):
    def construct(self):
        self.x_axis_label, self.y_axis_label = None, None
        self.setup_axes()
        func_graph = self.get_graph(
            lambda t: t % 1.0,
            x_min=-2 + EPSILON,
            x_max=2 - EPSILON,
            discontinuities=range(-2, 2),
        )
        self.play(Animation(func_graph))
        

MODULE_NAME = "plot"

@pytest.mark.slow
@pytest.mark.parametrize("scene_to_test", get_scenes_to_test(__name__), indirect=False)
def test_scene(scene_to_test, tmpdir, show_diff):
    GraphicalUnitTester(scene_to_test[1], MODULE_NAME, tmpdir).test(show_diff=show_diff)
